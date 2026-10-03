"""麥克風請求 schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = ["MicrophoneJobRequest", "MicrophoneDevice"]


class MicrophoneJobRequest(BaseModel):
    device_id: int | None = Field(default=None, description="麥克風裝置 id；空=預設輸入")


class MicrophoneDevice(BaseModel):
    id: int
    name: str
    sample_rate: int = 44100
    channels: int = 1
    host_api: str = ""
