"""本機影片服務：Embedded 文字字幕優先，否則 Whisper."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, Callable

from app.schemas.transcript import TranscriptResponse, TranscriptSegment
from app.services.errors import AppError
from app.services.media_range import clip_media_file, validate_and_clamp_range
from app.services.transcript import (
    adjust_whisper_timestamps,
    filter_and_clamp_existing_subtitles,
    to_srt,
    to_txt,
    to_vtt,
)
from app.services.youtube import parse_vtt_text

logger = logging.getLogger(__name__)

__all__ = [
    "SUPPORTED_VIDEO_EXTS",
    "TEXT_SUBTITLE_CODECS",
    "BITMAP_SUBTITLE_CODECS",
    "save_upload_file",
    "probe_media",
    "find_text_subtitle_stream",
    "extract_embedded_vtt",
    "process_video_file",
]

SUPPORTED_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".flv", ".wmv", ".mpeg", ".mpg"}

TEXT_SUBTITLE_CODECS = {
    "subrip", "srt", "ass", "ssa", "mov_text", "webvtt", "vtt", "ttml", "sami", "smi", "microdvd",
}

BITMAP_SUBTITLE_CODECS = {"dvd_subtitle", "dvb_subtitle", "hdmv_pgs_subtitle", "xsub", "dvdsub"}


def save_upload_file(upload_file: Any, dest_dir: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """chunk 寫入暫存，不整檔讀入 RAM；回傳 safe path."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    orig = getattr(upload_file, "filename", "") or "upload"
    ext = Path(orig).suffix.lower()
    safe = f"{uuid.uuid4().hex}{ext}"
    out = dest / safe
    try:
        with open(out, "wb") as f:
            while True:
                chunk = upload_file.file.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
    except Exception as exc:
        raise AppError("upload_storage_failed", str(exc)[:200]) from exc
    return str(out)


def probe_media(path: str | Path) -> dict[str, Any]:
    cmd: list[str] = [
        "ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", "-show_format", str(path),
    ]
    try:
        proc = subprocess.run(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except FileNotFoundError as exc:
        raise AppError("media_probe_failed", "ffprobe missing") from exc
    if proc.returncode != 0:
        raise AppError("media_probe_failed", (proc.stderr or b"").decode("utf-8", errors="ignore")[-300:])
    try:
        return json.loads((proc.stdout or b"").decode("utf-8", errors="ignore"))
    except Exception as exc:
        raise AppError("media_probe_failed", "invalid probe json") from exc


def _media_duration(probe: dict[str, Any]) -> float | None:
    try:
        fmt = probe.get("format") or {}
        if fmt.get("duration") is not None:
            return float(fmt["duration"])
    except Exception:
        pass
    try:
        for s in probe.get("streams", []):
            if s.get("duration") is not None:
                return float(s["duration"])
    except Exception:
        pass
    return None


def find_text_subtitle_stream(probe: dict[str, Any]) -> int | None:
    """回傳第一個可用文字字幕 stream index（於 subtitle 類別中的相對序號），無則 None."""
    idx = 0
    for s in probe.get("streams", []) or []:
        if (s.get("codec_type") or "").lower() != "subtitle":
            continue
        codec = (s.get("codec_name") or "").lower()
        if codec in BITMAP_SUBTITLE_CODECS:
            idx += 1
            continue
        if codec in TEXT_SUBTITLE_CODECS or codec == "":
            return idx
        idx += 1
    return None


def has_audio_stream(probe: dict[str, Any]) -> bool:
    for s in probe.get("streams", []) or []:
        if (s.get("codec_type") or "").lower() == "audio":
            return True
    return False


def extract_embedded_vtt(input_path: str | Path, output_vtt: str | Path, sub_index: int = 0) -> str:
    cmd: list[str] = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-map", f"0:s:{sub_index}", "-c:s", "webvtt", str(output_vtt),
    ]
    try:
        proc = subprocess.run(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except FileNotFoundError as exc:
        raise AppError("embedded_subtitle_extraction_failed", "ffmpeg missing") from exc
    if proc.returncode != 0:
        raise AppError("embedded_subtitle_extraction_failed", (proc.stderr or b"").decode("utf-8", errors="ignore")[-300:])
    if not Path(output_vtt).is_file():
        raise AppError("embedded_subtitle_extraction_failed", "no vtt produced")
    return str(output_vtt)


def process_video_file(
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
        raise AppError("invalid_video_file", "not found")
    probe = probe_media(src)
    duration = _media_duration(probe)
    if duration is None:
        raise AppError("media_probe_failed", "no duration")

    rs, re = validate_and_clamp_range(start_hm, end_hm, duration, is_default_end)

    tmp_paths: list[str] = []
    try:
        sub_idx = find_text_subtitle_stream(probe)
        if sub_idx is not None:
            vtt_tmp = str(Path(tempfile.gettempdir()) / f"tiger_emb_{uuid.uuid4().hex}.vtt")
            tmp_paths.append(vtt_tmp)
            extract_embedded_vtt(src, vtt_tmp, sub_idx)
            raw_text = Path(vtt_tmp).read_text(encoding="utf-8", errors="ignore")
            segments = parse_vtt_text(raw_text)
            # 語言：取 subtitle stream language
            lang: str | None = None
            try:
                n = 0
                for s in probe.get("streams", []):
                    if (s.get("codec_type") or "").lower() == "subtitle":
                        if n == sub_idx:
                            tags = s.get("tags") or {}
                            lang = tags.get("language")
                            break
                        n += 1
            except Exception:
                lang = None
            segments = filter_and_clamp_existing_subtitles(segments, rs, re)
            if progress_callback:
                for sg in segments:
                    try:
                        progress_callback({"start": sg.start, "end": sg.end, "text": sg.text})
                    except Exception:
                        pass
            return TranscriptResponse(
                language=lang, type="embedded", selection_mode="auto",
                segment_count=len(segments), duration=duration, segments=segments,
                vtt=to_vtt(segments), txt=to_txt(segments), srt=to_srt(segments),
                range_start=rs, range_end=re,
            )

        if not has_audio_stream(probe):
            raise AppError("no_transcript_source_available")

        transcribe_path = str(src)
        clip_tmp: str | None = None
        if rs > 0 or re < duration:
            clip_tmp = str(Path(tempfile.gettempdir()) / f"tiger_clip_{uuid.uuid4().hex}.mp4")
            tmp_paths.append(clip_tmp)
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
            type="transcribed", selection_mode="fallback",
            segment_count=len(segments), duration=duration, segments=segments,
            vtt=to_vtt(segments), txt=to_txt(segments), srt=to_srt(segments),
            range_start=rs, range_end=re,
            transcription_model=getattr(whisper_manager, "model_name", "large-v3"),
            device=getattr(whisper_manager, "working_device", None),
            compute_type=getattr(whisper_manager, "working_compute", None),
        )
    finally:
        for p in tmp_paths:
            try:
                if p and Path(p).is_file():
                    os.remove(p)
            except Exception:
                pass
