"""YouTube 來源服務：字幕優先、無字幕 Whisper fallback."""

from __future__ import annotations

import logging
import os
import re
import tempfile
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

logger = logging.getLogger(__name__)

__all__ = [
    "LANG_PRIORITY",
    "select_subtitle_track",
    "parse_vtt_text",
    "fetch_video_info",
    "download_subtitle_text",
    "download_bestaudio",
    "process_youtube_url",
    "build_response",
]

LANG_PRIORITY = ["zh-TW", "zh-CN", "en"]


def _normalize_lang(lang: str) -> str:
    return str(lang or "").strip().lower().replace("_", "-")


def select_subtitle_track(
    subtitles: dict[str, Any] | None, automatic_captions: dict[str, Any] | None
) -> dict[str, Any] | None:
    """deterministic 選擇：manual 優先於 auto；語言 zh-TW -> zh-CN -> en."""
    subs = subtitles or {}
    autos = automatic_captions or {}
    # 建立正規化索引以支援大小寫 / 底線差異
    def _index(d: dict[str, Any]) -> dict[str, str]:
        idx: dict[str, str] = {}
        for k in d.keys():
            idx[_normalize_lang(k)] = k
        return idx

    sub_idx = _index(subs)
    auto_idx = _index(autos)
    for lang in LANG_PRIORITY:
        nl = _normalize_lang(lang)
        if nl in sub_idx:
            key = sub_idx[nl]
            tracks = subs.get(key) or []
            if tracks:
                t = tracks[0]
                return {"lang": key, "ext": t.get("ext", "vtt"), "url": t.get("url"), "is_auto": False}
        # 前綴相容：如 "zh" 命中 "zh-TW"？僅在完全無精確命中時考慮，此處略過以保持精確。
    for lang in LANG_PRIORITY:
        nl = _normalize_lang(lang)
        if nl in auto_idx:
            key = auto_idx[nl]
            tracks = autos.get(key) or []
            if tracks:
                t = tracks[0]
                return {"lang": key, "ext": t.get("ext", "vtt"), "url": t.get("url"), "is_auto": True}
    # fallback：任一 manual 軌
    for key, tracks in subs.items():
        if tracks:
            t = tracks[0]
            return {"lang": key, "ext": t.get("ext", "vtt"), "url": t.get("url"), "is_auto": False}
    for key, tracks in autos.items():
        if tracks:
            t = tracks[0]
            return {"lang": key, "ext": t.get("ext", "vtt"), "url": t.get("url"), "is_auto": True}
    return None


_TS_RE = re.compile(r"(\d+):(\d{2}):(\d{2})[.,](\d{3})")


def _parse_ts(ts: str) -> float:
    m = _TS_RE.search(ts.strip())
    if not m:
        raise ValueError(f"bad timestamp: {ts}")
    h, mi, s, ms = (int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))
    return float(h * 3600 + mi * 60 + s + ms / 1000.0)


_TAG_RE = re.compile(r"<[^>]+>")


def _clean_text(text: str) -> str:
    t = _TAG_RE.sub("", text or "")
    t = t.replace("&nbsp;", " ").strip()
    return " ".join(t.split())


def parse_vtt_text(vtt_text: str) -> list[TranscriptSegment]:
    """解析 VTT 全文為 Canonical segments；YouTube rolling 重複做輕量去重."""
    text = (vtt_text or "").replace("\r\n", "\n").replace("\r", "\n")
    # 移除 WEBVTT 標頭行與 NOTE/STYLE 區塊
    blocks = re.split(r"\n\s*\n", text)
    segments: list[TranscriptSegment] = []
    prev_text = ""
    for block in blocks:
        lines = [ln.strip() for ln in block.strip().split("\n") if ln.strip() != ""]
        if not lines:
            continue
        if lines[0].upper().startswith("WEBVTT"):
            continue
        if lines[0].upper().startswith("NOTE") or lines[0].upper().startswith("STYLE"):
            continue
        # 找到時間軸行
        ts_idx = -1
        for i, ln in enumerate(lines):
            if "-->" in ln:
                ts_idx = i
                break
        if ts_idx < 0:
            continue
        try:
            start_s, end_s = lines[ts_idx].split("-->", 1)
            start = _parse_ts(start_s)
            end = _parse_ts(end_s)
        except Exception:
            continue
        cue = _clean_text(" ".join(lines[ts_idx + 1 :]))
        if not cue:
            continue
        # rolling-caption 輕量去重：與上一 cue 完全相同則跳過
        if cue == prev_text:
            continue
        # 若新 cue 完整包含上一 cue 且時間相鄰，僅保留新 cue 文字（避免合法重複語音被误删：
        # Whisper 不套用此規則，此處僅 YouTube 字幕）
        segments.append(TranscriptSegment(start=start, end=end, text=cue))
        prev_text = cue
    segments.sort(key=lambda x: (x.start, x.end))
    return segments


def fetch_video_info(url: str) -> dict[str, Any]:
    try:
        import yt_dlp  # type: ignore
    except Exception as exc:
        raise AppError("audio_download_failed", f"yt-dlp missing: {exc}") from exc
    ydl_opts: dict[str, Any] = {"quiet": True, "no_warnings": True, "skip_download": True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except AppError:
        raise
    except Exception as exc:
        raise AppError("audio_download_failed", str(exc)[:300]) from exc
    if not isinstance(info, dict):
        raise AppError("audio_download_failed", "empty info")
    return info


def download_subtitle_text(track_url: str) -> str:
    if not track_url:
        raise AppError("subtitle_track_not_found")
    try:
        import urllib.request

        req = urllib.request.Request(track_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
        return raw.decode("utf-8", errors="ignore")
    except AppError:
        raise
    except Exception as exc:
        raise AppError("subtitle_track_not_found", str(exc)[:200]) from exc


def _resolve_downloaded_file(outtmpl: str | Path) -> str | None:
    """解析 yt-dlp 實際落地檔。

    outtmpl 通常帶 `%(ext)s` 樣板（副檔名由 yt-dlp 決定，如 .webm/.m4a），
    因此絕不能直接拿樣板字串做 is_file；需以後綴前的前綴做 glob 尋找。
    """
    p = Path(str(outtmpl))
    if p.is_file():
        return str(p)
    name = p.name
    parent = p.parent
    if "%(ext)s" in name:
        prefix = name.split("%(ext)s")[0].rstrip(".")
        pattern = (prefix + ".*") if prefix else "*"
    else:
        # 純路徑：yt-dlp 可能自動追加 .ext
        pattern = name + ".*"
    try:
        cands = [c for c in parent.glob(pattern) if c.is_file()]
    except Exception:
        return None
    if not cands:
        return None
    # 同前綴多檔時取最大者（最可能是完整音訊，而非碎片）
    def _size(c: Path) -> int:
        try:
            return c.stat().st_size
        except Exception:
            return -1

    cands.sort(key=_size, reverse=True)
    return str(cands[0])


def download_bestaudio(url: str, out_path: str | Path) -> str:
    try:
        import yt_dlp  # type: ignore
    except Exception as exc:
        raise AppError("audio_download_failed", f"yt-dlp missing: {exc}") from exc
    # 包容性格式：有獨立 bestaudio 用之，否則退回含音訊的最佳混合格式；
    # 不設 format_limit，不限協定；noplaylist 避免整清單誤下載。
    ydl_opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "format": "bestaudio/best",
        "outtmpl": str(out_path),
        "noplaylist": True,
        "retries": 3,
        "fragment_retries": 3,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except AppError:
        raise
    except Exception as exc:
        # 含 yt_dlp.utils.DownloadError：串流切換/網路抖動一律轉受控錯誤
        raise AppError("audio_download_failed", str(exc)[:300]) from exc
    resolved = _resolve_downloaded_file(out_path)
    if resolved is None:
        raise AppError("audio_download_failed", "bestaudio not found")
    return resolved


def build_response(
    segments: list[TranscriptSegment],
    language: str | None,
    type_: str,
    selection_mode: str,
    duration: float | None = None,
    range_start: float | None = None,
    range_end: float | None = None,
    extra: dict[str, Any] | None = None,
) -> TranscriptResponse:
    segs = sorted(segments, key=lambda x: (x.start, x.end))
    base: dict[str, Any] = {
        "language": language,
        "type": type_,
        "selection_mode": selection_mode,
        "segment_count": len(segs),
        "duration": duration,
        "segments": segs,
        "vtt": to_vtt(segs),
        "txt": to_txt(segs),
        "srt": to_srt(segs),
        "range_start": range_start,
        "range_end": range_end,
    }
    if extra:
        base.update(extra)
    return TranscriptResponse(**base)


def _is_stopped(stop_checker: Callable[[], bool] | None) -> bool:
    if stop_checker is None:
        return False
    try:
        return bool(stop_checker())
    except Exception:
        return False


def _transcribe_youtube_audio(
    url: str,
    duration: float | None,
    rs: float | None,
    re: float | None,
    start_hm: str | None,
    end_hm: str | None,
    is_default_end: bool,
    stop_checker: Callable[[], bool] | None,
    progress_callback: Callable[[Any], Any] | None,
    whisper_manager: Any | None,
) -> TranscriptResponse:
    """Whisper ASR 慢軌：下載 audio-only → 區間裁切 → 本機轉錄（受 STOP 監控）."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="tigeryt_"))
    audio_path = tmp_dir / "audio.%(ext)s"
    downloaded: str | None = None
    clip_path: str | None = None
    try:
        downloaded = download_bestaudio(url, str(audio_path))
        if _is_stopped(stop_checker):
            return build_response([], None, "transcribed", "fallback", duration, None, None)
        transcribe_path = downloaded
        if duration is not None and duration > 0:
            if rs is None or re is None:
                rs, re = validate_and_clamp_range(start_hm, end_hm, duration, is_default_end)
            if rs > 0 or re < duration:
                clip_path = str(tmp_dir / "clip.mp4")
                clip_media_file(downloaded, clip_path, rs, re)
                transcribe_path = clip_path
        if whisper_manager is None:
            from app.services.whisper import get_whisper_manager

            whisper_manager = get_whisper_manager()

        offset = float(rs) if rs else 0.0

        def _cb(item: Any) -> Any:
            d = dict(item) if isinstance(item, dict) else {"start": item.get("start"), "end": item.get("end"), "text": item.get("text")}
            try:
                d["start"] = float(d.get("start", 0.0)) + offset
                d["end"] = float(d.get("end", d.get("start", 0.0))) + offset
            except Exception:
                pass
            if progress_callback is not None:
                return progress_callback(d)
            return None

        raw = whisper_manager.transcribe_file(transcribe_path, vad_filter=True, callback=_cb if progress_callback else None, stop_checker=stop_checker)
        # transcribe 回傳 clip-relative；callback 已 offset 報給 job，此處統一 offset 回傳值
        segments = adjust_whisper_timestamps(
            [TranscriptSegment(start=float(s.get("start", 0.0)), end=float(s.get("end", 0.0)), text=str(s.get("text", ""))) for s in raw],
            offset,
        )
        lang = None
        try:
            lang = getattr(whisper_manager, "_last_language", None)
        except Exception:
            lang = None
        if lang is not None and not isinstance(lang, str):
            lang = None
        _device = getattr(whisper_manager, "working_device", None)
        _compute = getattr(whisper_manager, "working_compute", None)
        _model = getattr(whisper_manager, "model_name", "large-v3")
        extra = {
            "transcription_model": _model if isinstance(_model, str) else "large-v3",
            "device": _device if isinstance(_device, str) else None,
            "compute_type": _compute if isinstance(_compute, str) else None,
        }
        return build_response(segments, lang, "transcribed", "fallback", duration, rs, re, extra)
    finally:
        for p in (downloaded, clip_path):
            try:
                if p and Path(p).is_file():
                    os.remove(p)
            except Exception:
                pass
        try:
            # 清理 glob 產生的 bestaudio（副檔名不確定時）
            for cand in tmp_dir.glob("audio.*"):
                try:
                    if cand.is_file():
                        os.remove(cand)
                except Exception:
                    pass
            tmp_dir.rmdir()
        except Exception:
            pass


def process_youtube_url(
    url: str,
    start_hm: str | None = "0:0",
    end_hm: str | None = "0:10",
    is_default_end: bool = True,
    job_id: str | None = None,
    stop_checker: Callable[[], bool] | None = None,
    progress_callback: Callable[[Any], Any] | None = None,
    whisper_manager: Any | None = None,
    duration_override: float | None = None,
) -> TranscriptResponse:
    info = fetch_video_info(url)
    duration: float | None = duration_override
    if duration is None:
        try:
            duration = float(info.get("duration")) if info.get("duration") is not None else None
        except Exception:
            duration = None

    # 區間先驗證（失敗直接拋 invalid_time_range，不進任何下載）
    rs: float | None = None
    re: float | None = None
    if duration is not None and duration > 0:
        rs, re = validate_and_clamp_range(start_hm, end_hm, duration, is_default_end)
    elif duration is not None:
        rs, re = (0.0, duration)
    else:
        rs, re = (None, None)  # type: ignore

    # Fast path：文字字幕優先
    track = select_subtitle_track(info.get("subtitles"), info.get("automatic_captions"))
    if track is not None and track.get("url"):
        try:
            vtt_text = download_subtitle_text(track["url"])
            segments = parse_vtt_text(vtt_text)
            if duration is not None and duration > 0 and rs is not None and re is not None:
                segments = filter_and_clamp_existing_subtitles(segments, rs, re)
        except AppError:
            # 字幕下載/解析失敗 → 視為無有效字幕，強制走 Whisper 慢軌
            logger.warning("YouTube subtitle fast path failed; falling back to Whisper ASR")
            segments = []
        if len(segments) > 0:
            lang = str(track.get("lang") or info.get("language") or "en")
            type_ = "auto" if track.get("is_auto") else "manual"
            if progress_callback is not None:
                for s in segments:
                    try:
                        progress_callback({"start": s.start, "end": s.end, "text": s.text})
                    except Exception:
                        pass
            return build_response(segments, lang, type_, "auto", duration, rs, re)
        # 該區間內無實質有效文字 → 絕對不能直接結束，強制 Fallback 進 Whisper
        logger.info("YouTube subtitle track empty after range filter; falling back to Whisper ASR")
        if _is_stopped(stop_checker):
            return build_response([], None, "transcribed", "fallback", duration, rs, re)

    # 慢軌：無字幕軌，或快軌 0 segments → bestaudio + Whisper（同受 Job/STOP 監控）
    return _transcribe_youtube_audio(
        url, duration, rs, re, start_hm, end_hm, is_default_end,
        stop_checker, progress_callback, whisper_manager,
    )
