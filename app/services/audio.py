"""本機語音檔服務：FFprobe 驗證後直轉 Whisper."""

from __future__ import annotations

import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Callable

from app.schemas.transcript import TranscriptResponse, TranscriptSegment
from app.services.errors import AppError
from app.services.media_range import clip_media_file, validate_and_clamp_range
from app.services.transcript import adjust_whisper_timestamps, to_srt, to_txt, to_vtt
from app.services.video import has_audio_stream, probe_media, save_upload_file

logger = logging.getLogger(__name__)

__all__ = ["SUPPORTED_AUDIO_EXTS", "has_video_stream", "process_audio_file"]

SUPPORTED_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma"}


def has_video_stream(probe: dict[str, Any]) -> bool:
    for s in probe.get("streams", []) or []:
        if (s.get("codec_type") or "").lower() == "video":
            return True
    return False


def process_audio_file(
    path: str | Path,
    start_hm: str | None = "0:0",
    end_hm: str | None = "0:10",
    is_default_end: bool = True,
    job_id: str | None = None,
    stop_checker: Callable[[], bool] | None = None,
    progress_callback: Callable[[Any], Any] | None = None,
    whisper_manager: Any | None = None,
) -> TranscriptResponse:
    src = Path(path)
    if not src.is_file():
        raise AppError("invalid_audio_file", "not found")
    probe = probe_media(src)
    if has_video_stream(probe):
        raise AppError("unsupported_audio_format", "video stream present; use video endpoint")
    if not has_audio_stream(probe):
        raise AppError("invalid_audio_file", "no audio stream")

    duration: float | None = None
    try:
        fmt = probe.get("format") or {}
        duration = float(fmt["duration"]) if fmt.get("duration") is not None else None
    except Exception:
        duration = None
    if duration is None:
        raise AppError("media_probe_failed", "no duration")

    rs, re = validate_and_clamp_range(start_hm, end_hm, duration, is_default_end)

    clip_tmp: str | None = None
    try:
        transcribe_path = str(src)
        if rs > 0 or re < duration:
            clip_tmp = str(Path(tempfile.gettempdir()) / f"tiger_aclip_{uuid.uuid4().hex}{src.suffix or '.mp4'}")
            clip_media_file(src, clip_tmp, rs, re)
            transcribe_path = clip_tmp

        if whisper_manager is None:
            from app.services.whisper import get_whisper_manager

            whisper_manager = get_whisper_manager()

        def _cb(item: Any) -> Any:
            d = dict(item) if isinstance(item, dict) else {"start": item.get("start"), "end": item.get("end"), "text": item.get("text")}
            try:
                d["start"] = float(d.get("start", 0.0)) + rs
                d["end"] = float(d.get("end", d.get("start", 0.0))) + rs
            except Exception:
                pass
            if progress_callback is not None:
                return progress_callback(d)
            return None

        raw = whisper_manager.transcribe_file(transcribe_path, vad_filter=True, callback=_cb if progress_callback else None, stop_checker=stop_checker)
        segments = adjust_whisper_timestamps(
            [TranscriptSegment(start=float(s.get("start", 0.0)), end=float(s.get("end", 0.0)), text=str(s.get("text", ""))) for s in raw],
            rs,
        )
        return TranscriptResponse(
            language=getattr(whisper_manager, "_last_language", None),
            type="transcribed", selection_mode="direct",
            segment_count=len(segments), duration=duration, segments=segments,
            vtt=to_vtt(segments), txt=to_txt(segments), srt=to_srt(segments),
            range_start=rs, range_end=re,
            transcription_model=getattr(whisper_manager, "model_name", "large-v3"),
            device=getattr(whisper_manager, "working_device", None),
            compute_type=getattr(whisper_manager, "working_compute", None),
        )
    finally:
        if clip_tmp:
            try:
                if Path(clip_tmp).is_file():
                    os.remove(clip_tmp)
            except Exception:
                pass
