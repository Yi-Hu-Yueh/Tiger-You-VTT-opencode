"""YouTube 請求 schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = ["YoutubeInfoRequest", "YoutubeSubtitleRequest", "YoutubeJobRequest"]


class YoutubeInfoRequest(BaseModel):
    url: str = Field(description="YouTube 影片 URL")


class YoutubeSubtitleRequest(BaseModel):
    url: str = Field(description="YouTube 影片 URL")
    start_time: str | None = Field(default="0:0", description="H:M，空白=從頭")
    end_time: str | None = Field(default="0:10", description="H:M，空白=到結尾")


class YoutubeJobRequest(YoutubeSubtitleRequest):
    pass
