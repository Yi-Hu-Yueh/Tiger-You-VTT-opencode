"""Canonical transcript schemas.

所有來源（YouTube / 本機影片 / 語音檔 / Windows 系統音訊）
最終統一為 TranscriptSegment[]，再序列化為 TXT / VTT / SRT。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = ["TranscriptSegment", "TranscriptResponse"]


class TranscriptSegment(BaseModel):
    """Canonical 單一片段：使用浮點秒數時間軸。"""

    start: float = Field(description="片段開始時間（秒）")
    end: float = Field(description="片段結束時間（秒）")
    text: str = Field(description="清理後字幕 / ASR 文本")


class TranscriptResponse(BaseModel):
    """共通轉錄回應：有限媒體用原影片 timeline，系統音訊用 session-relative timeline。"""

    language: str | None = Field(default=None, description="字幕軌語言或 Whisper 偵測語言")
    type: str = Field(default="transcribed", description="manual / auto / embedded / transcribed")
    selection_mode: str = Field(
        default="direct", description="auto / explicit / fallback / direct"
    )
    segment_count: int = Field(default=0, ge=0, description="segments 數量")
    duration: float | None = Field(default=None, description="媒體或 session 長度（秒）")
    segments: list[TranscriptSegment] = Field(default_factory=list)
    vtt: str = Field(default="", description="WebVTT 全文")
    txt: str = Field(default="", description="逗號串接純文字")
    srt: str = Field(default="", description="SRT 全文")

    # Whisper / range / live metadata（可選，不影響 canonical 核心）
    transcription_model: str | None = Field(default=None)
    device: str | None = Field(default=None, description="實際推論裝置：cuda / cpu")
    compute_type: str | None = Field(default=None, description="實際 compute type")
    range_start: float | None = Field(default=None, description="H:M 區間起點（秒）")
    range_end: float | None = Field(default=None, description="H:M 區間終點（秒）")
