"""系統音訊請求 schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = ["SystemAudioJobRequest", "SystemAudioDevice"]


class SystemAudioJobRequest(BaseModel):
    device_id: int | None = Field(default=None, description="WASAPI loopback 裝置 id；空=預設輸出")


class SystemAudioDevice(BaseModel):
    id: int
    name: str
    sample_rate: int = 48000
    channels: int = 2
    host_api: str = ""
