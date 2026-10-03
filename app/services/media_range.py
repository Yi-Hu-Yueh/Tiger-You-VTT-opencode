"""時間區間（H:M）控制與 FFmpeg 裁切.

- 使用者輸入固定為「小時:分鐘」，無秒欄位；內部一律轉 float 秒數。
- 現成字幕採 overlap 過濾 + 邊界 clamp；Whisper 採 clip + offset 回原 timeline。
- 所有外部程序必須使用 argument array 且 shell=False（Windows 安全）。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

__all__ = ["parse_hm_to_seconds", "validate_and_clamp_range", "clip_media_file"]

_DEFAULT_END_HM = "0:10"
_DEFAULT_END_SECONDS = 600.0


def parse_hm_to_seconds(hm_str: str | None) -> float | None:
    """解析 H:M 為秒數；空/空白回傳 None；格式不符拋 ValueError("invalid_time_range")."""
    if hm_str is None:
        return None
    s = str(hm_str).strip()
    if s == "":
        return None
    parts = s.split(":")
    if len(parts) != 2:
        raise ValueError("invalid_time_range")
    h_raw, m_raw = parts[0].strip(), parts[1].strip()
    if h_raw == "" or m_raw == "":
        raise ValueError("invalid_time_range")
    # 僅允許非負整數（無符號、無小數）
    if not h_raw.isdigit() or not m_raw.isdigit():
        raise ValueError("invalid_time_range")
    hours = int(h_raw)
    minutes = int(m_raw)
    if minutes < 0 or minutes > 59:
        raise ValueError("invalid_time_range")
    return float(hours * 3600 + minutes * 60)


def validate_and_clamp_range(
    start_hm: str | None,
    end_hm: str | None,
    duration: float,
    is_default_end: bool = False,
) -> tuple[float, float]:
    """驗證 H:M 區間並回傳 (start_sec, end_sec).

    - 開始留空 -> 0.0；結束留空 -> duration。
    - start < 0 / end < 0 / start >= end -> ValueError("invalid_time_range")。
    - 超出 duration 一律 ValueError，唯獨 untouched 預設 "0:10" 允許收斂到 duration。
    """
    try:
        dur = float(duration)
    except Exception:
        raise ValueError("invalid_time_range") from None

    start_sec = parse_hm_to_seconds(start_hm)
    end_sec = parse_hm_to_seconds(end_hm)
    if start_sec is None:
        start_sec = 0.0
    if end_sec is None:
        end_sec = dur

    if start_sec < 0 or end_sec < 0 or start_sec >= end_sec:
        raise ValueError("invalid_time_range")

    # 結束超出媒體長度：僅 untouched 預設情境可 clamp（以旗標區分手動輸入）
    if end_sec > dur:
        if bool(is_default_end):
            end_sec = dur
            if start_sec >= end_sec:
                raise ValueError("invalid_time_range")
            return (float(start_sec), float(end_sec))
        raise ValueError("invalid_time_range")

    if start_sec > dur:
        raise ValueError("invalid_time_range")

    return (float(start_sec), float(end_sec))


def clip_media_file(input_path: str | Path, output_path: str | Path, start_sec: float, end_sec: float) -> str:
    """以 FFmpeg 裁切 [start_sec, end_sec) 區間；回傳 output_path 字串."""
    try:
        start = float(start_sec)
        end = float(end_sec)
    except Exception:
        raise ValueError("invalid_time_range") from None
    if start < 0 or end < 0 or start >= end:
        raise ValueError("invalid_time_range")

    src = Path(input_path)
    dst = Path(output_path)
    if not src.is_file():
        raise FileNotFoundError(f"media file not found: {src}")

    cmd: list[str] = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start),
        "-to",
        str(end),
        "-i",
        str(src),
        "-c",
        "copy",
        str(dst),
    ]
    proc = subprocess.run(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        err = ""
        try:
            err = (proc.stderr or b"").decode("utf-8", errors="ignore")[-500:]
        except Exception:
            err = ""
        raise RuntimeError(f"range_clip_failed: ffmpeg exit {proc.returncode} {err}")
    return str(dst)
