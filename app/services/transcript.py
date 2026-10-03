"""Canonical 字幕序列化與時間軸修正.

- 所有來源統一為 TranscriptSegment[] 後，再經此序列化為 TXT / VTT / SRT。
- Whisper clip 相對時間經 offset 加回原 timeline；現成字幕採 overlap + clamp。
"""

from __future__ import annotations

from typing import Any, Sequence

from app.schemas.transcript import TranscriptSegment

__all__ = [
    "to_txt",
    "to_vtt",
    "to_srt",
    "adjust_whisper_timestamps",
    "filter_and_clamp_existing_subtitles",
]


def _coerce(segment: TranscriptSegment | dict[str, Any]) -> TranscriptSegment:
    if isinstance(segment, TranscriptSegment):
        return segment
    if isinstance(segment, dict):
        return TranscriptSegment(
            start=float(segment.get("start", 0.0)),
            end=float(segment.get("end", segment.get("start", 0.0))),
            text=str(segment.get("text", "") or ""),
        )
    return TranscriptSegment(
        start=float(getattr(segment, "start", 0.0)),
        end=float(getattr(segment, "end", getattr(segment, "start", 0.0))),
        text=str(getattr(segment, "text", "") or ""),
    )


def to_txt(segments: Sequence[TranscriptSegment | dict[str, Any]]) -> str:
    """不同 segment 以 ASCII 半形逗號串接；末尾不加多餘逗號."""
    return ",".join(_coerce(s).text for s in segments)


def _split_ms(total_seconds: float) -> tuple[int, int, int, int]:
    total_ms = int(round(float(total_seconds) * 1000.0))
    if total_ms < 0:
        total_ms = 0
    hours = total_ms // 3_600_000
    rem = total_ms % 3_600_000
    minutes = rem // 60_000
    rem = rem % 60_000
    seconds = rem // 1000
    millis = rem % 1000
    return hours, minutes, seconds, millis


def _format_vtt_ts(seconds: float) -> str:
    h, m, s, ms = _split_ms(seconds)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _format_srt_ts(seconds: float) -> str:
    h, m, s, ms = _split_ms(seconds)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_vtt(segments: Sequence[TranscriptSegment | dict[str, Any]]) -> str:
    """WEBVTT + HH:MM:SS.mmm 毫秒精度."""
    lines: list[str] = ["WEBVTT", ""]
    for seg in segments:
        s = _coerce(seg)
        lines.append(f"{_format_vtt_ts(s.start)} --> {_format_vtt_ts(s.end)}")
        lines.append(s.text)
        lines.append("")
    # 空列表時僅回傳標頭；非空時維持每 cue 後空行
    if len(segments) == 0:
        return "WEBVTT\n"
    return "\n".join(lines) + "\n"


def to_srt(segments: Sequence[TranscriptSegment | dict[str, Any]]) -> str:
    """連續序號（1 起）+ HH:MM:SS,mmm."""
    if len(segments) == 0:
        return ""
    lines: list[str] = []
    for i, seg in enumerate(segments, start=1):
        s = _coerce(seg)
        lines.append(str(i))
        lines.append(f"{_format_srt_ts(s.start)} --> {_format_srt_ts(s.end)}")
        lines.append(s.text)
        lines.append("")
    return "\n".join(lines) + "\n"


def adjust_whisper_timestamps(
    segments: Sequence[TranscriptSegment | dict[str, Any]], range_start: float
) -> list[TranscriptSegment]:
    """將 clip-relative 時間戳統一加回 range_start，回到原媒體 timeline."""
    try:
        offset = float(range_start)
    except Exception:
        offset = 0.0
    out: list[TranscriptSegment] = []
    for seg in segments:
        s = _coerce(seg)
        out.append(TranscriptSegment(start=s.start + offset, end=s.end + offset, text=s.text))
    return out


def filter_and_clamp_existing_subtitles(
    segments: Sequence[TranscriptSegment | dict[str, Any]], range_start: float, range_end: float
) -> list[TranscriptSegment]:
    """保留與 [range_start, range_end) 有重疊的 cue，並 clamp 至邊界；不建立內部新 timestamp."""
    try:
        rs = float(range_start)
        re = float(range_end)
    except Exception:
        raise ValueError("invalid_time_range") from None
    if rs < 0 or re < 0 or rs >= re:
        raise ValueError("invalid_time_range")
    out: list[TranscriptSegment] = []
    for seg in segments:
        s = _coerce(seg)
        # 無重疊則丟棄（含剛好相鄰不重疊）
        if s.end <= rs or s.start >= re:
            continue
        ns = s.start if s.start > rs else rs
        ne = s.end if s.end < re else re
        if ns >= ne:
            continue
        out.append(TranscriptSegment(start=ns, end=ne, text=s.text))
    out.sort(key=lambda x: (x.start, x.end))
    return out
