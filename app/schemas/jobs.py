"""Job 狀態 schemas：背景 Job 輪詢（Polling）契約."""

from __future__ import annotations

from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field

from app.schemas.transcript import TranscriptSegment

__all__ = ["JobStatus", "JobStatusResponse"]


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    COMPLETED = "completed"
    FAILED = "failed"


class JobStatusResponse(BaseModel):
    """GET /api/jobs/{job_id} 輪詢回應；elapsed 凍結語意由 JobService 保證."""

    job_id: UUID = Field(description="UUID4 Job 識別碼")
    source_type: str = Field(description="youtube / video / audio / system_audio")
    status: JobStatus = Field(description="背景 Job 狀態機")
    elapsed_seconds: float = Field(ge=0.0, description="自建立起算的 monotonic 秒數，終端狀態凍結")
    segment_count: int = Field(default=0, ge=0)
    current_segments: list[TranscriptSegment] = Field(default_factory=list)
    current_txt: str = Field(default="", description="半形逗號串接的目前文字")

    error: str | None = Field(default=None, description="failed 時的 client-safe 錯誤")
