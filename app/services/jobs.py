"""背景 Job 控制核心.

- 單執行緒 ThreadPoolExecutor(max_workers=1)，避免 large-v3 資源競爭。
- In-memory registry，重啟不持久化。
- 權威計時：嚴格基於 time.monotonic()；建立(queued)起算，終端凍結。
- Cooperative STOP：threading.Event + 安全邊界跳出，保留 partial。
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable

from app.schemas.jobs import JobStatus
from app.schemas.transcript import TranscriptSegment

logger = logging.getLogger(__name__)

__all__ = ["JobService", "job_service", "get_job_service"]

_TERMINAL_STATUSES = (JobStatus.STOPPED, JobStatus.COMPLETED, JobStatus.FAILED)


class JobService:
    def __init__(self, max_workers: int = 1) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    # -- 相容別名 ---------------------------------------------------------
    @property
    def executor(self) -> ThreadPoolExecutor:
        return self._executor

    @property
    def jobs(self) -> dict[str, dict[str, Any]]:
        return self._jobs

    # -- 建立 / 查詢 -------------------------------------------------------
    def create_job(
        self,
        source_type: str,
        range_start: float | None = None,
        range_end: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        job_id = str(uuid.uuid4())
        now = time.monotonic()
        record: dict[str, Any] = {
            "job_id": job_id,
            "source_type": source_type,
            "status": JobStatus.QUEUED,
            "start_monotonic": now,
            "end_monotonic": None,
            "stop_event": threading.Event(),
            "segments": [],
            "current_txt": "",
            "error": None,
            "range_start": range_start,
            "range_end": range_end,
            "metadata": dict(metadata or {}),
            "future": None,
        }
        with self._lock:
            self._jobs[job_id] = record
        return self.get_status(job_id)

    def _get_or_raise(self, job_id: str) -> dict[str, Any]:
        try:
            return self._jobs[str(job_id)]
        except KeyError:
            raise KeyError(f"job_not_found: {job_id}") from None

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        return self._jobs.get(str(job_id))

    def _elapsed_locked(self, record: dict[str, Any]) -> float:
        start = record["start_monotonic"]
        end = record.get("end_monotonic")
        if end is not None:
            return max(0.0, end - start)
        return max(0.0, time.monotonic() - start)

    def get_status(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._get_or_raise(job_id)
            segments = list(record["segments"])
            return {
                "job_id": record["job_id"],
                "source_type": record["source_type"],
                "status": record["status"],
                "elapsed_seconds": self._elapsed_locked(record),
                "segment_count": len(segments),
                "current_segments": segments,
                "current_txt": record.get("current_txt", ""),
                "error": record.get("error"),
            }

    # -- 終端凍結 ----------------------------------------------------------
    def _set_terminal_locked(self, record: dict[str, Any], status: JobStatus, error: str | None = None) -> None:
        if record["status"] in _TERMINAL_STATUSES:
            return
        record["status"] = status
        if record.get("end_monotonic") is None:
            record["end_monotonic"] = time.monotonic()
        if error is not None:
            record["error"] = error

    def _set_terminal(self, job_id: str, status: JobStatus, error: str | None = None) -> None:
        with self._lock:
            record = self._get_or_raise(job_id)
            self._set_terminal_locked(record, status, error)

    # -- Cooperative STOP --------------------------------------------------
    def stop_job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._get_or_raise(job_id)
            if record["status"] in _TERMINAL_STATUSES:
                return self._status_locked(record)
            record["stop_event"].set()
            if record["status"] in (JobStatus.QUEUED, JobStatus.RUNNING):
                record["status"] = JobStatus.STOPPING
            return self._status_locked(record)

    def _status_locked(self, record: dict[str, Any]) -> dict[str, Any]:
        segments = list(record["segments"])
        return {
            "job_id": record["job_id"],
            "source_type": record["source_type"],
            "status": record["status"],
            "elapsed_seconds": self._elapsed_locked(record),
            "segment_count": len(segments),
            "current_segments": segments,
            "current_txt": record.get("current_txt", ""),
            "error": record.get("error"),
        }

    def is_stop_requested(self, job_id: str) -> bool:
        record = self._jobs.get(str(job_id))
        if record is None:
            return False
        try:
            return bool(record["stop_event"].is_set())
        except Exception:
            return False

    def stop_checker(self, job_id: str) -> bool:
        return self.is_stop_requested(job_id)

    # -- 進度回報 ------------------------------------------------------------
    @staticmethod
    def _coerce_segment(segment: TranscriptSegment | dict[str, Any]) -> TranscriptSegment:
        if isinstance(segment, TranscriptSegment):
            return segment
        if isinstance(segment, dict):
            return TranscriptSegment(
                start=float(segment.get("start", 0.0)),
                end=float(segment.get("end", segment.get("start", 0.0))),
                text=str(segment.get("text", "") or "").strip(),
            )
        start = float(getattr(segment, "start", 0.0))
        end = float(getattr(segment, "end", start))
        text = str(getattr(segment, "text", "") or "").strip()
        return TranscriptSegment(start=start, end=end, text=text)

    def update_progress_callback(self, job_id: str, segment: TranscriptSegment | dict[str, Any]) -> None:
        seg = self._coerce_segment(segment)
        if not seg.text:
            return
        with self._lock:
            record = self._get_or_raise(job_id)
            if record["status"] in _TERMINAL_STATUSES:
                return
            record["segments"].append(seg)
            record["current_txt"] = ",".join(s.text for s in record["segments"])

    # 別名：相容不同呼叫風格
    update_progress = update_progress_callback

    def callback_for(self, job_id: str) -> Callable[[Any], None]:
        def _cb(segment: Any) -> None:
            self.update_progress_callback(job_id, segment)

        return _cb

    make_callback = callback_for

    # -- 背景執行 ------------------------------------------------------------
    def submit(self, job_id: str, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
        with self._lock:
            self._get_or_raise(job_id)
        future: Future = self._executor.submit(self._run_wrapper, job_id, func, args, kwargs)
        with self._lock:
            self._jobs[str(job_id)]["future"] = future
        return future

    # 別名
    submit_job = submit
    run_in_background = submit
    enqueue = submit

    def create_and_submit(
        self,
        source_type: str,
        func: Callable[..., Any],
        *args: Any,
        range_start: float | None = None,
        range_end: float | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        created = self.create_job(source_type, range_start, range_end, metadata)
        self.submit(created["job_id"], func, *args, **kwargs)
        return created

    def _run_wrapper(self, job_id: str, func: Callable[..., Any], args: tuple, kwargs: dict) -> Any:
        with self._lock:
            record = self._get_or_raise(job_id)
            if record["status"] == JobStatus.QUEUED:
                if record["stop_event"].is_set():
                    record["status"] = JobStatus.STOPPING
                else:
                    record["status"] = JobStatus.RUNNING
        try:
            result = func(*args, **kwargs)
        except Exception as exc:
            logger.exception("Job %s failed: %s", job_id, exc)
            self._set_terminal(job_id, JobStatus.FAILED, error=str(exc) or "failed")
            raise
        # 正常返回：若期間被要求 STOP，轉 stopped 並保留 partial
        if self.is_stop_requested(job_id):
            self._set_terminal(job_id, JobStatus.STOPPED)
        else:
            self._set_terminal(job_id, JobStatus.COMPLETED)
        return result

    # -- 維護 ------------------------------------------------------------------
    def complete_job(self, job_id: str) -> dict[str, Any]:
        self._set_terminal(job_id, JobStatus.COMPLETED)
        return self.get_status(job_id)

    def fail_job(self, job_id: str, error: str) -> dict[str, Any]:
        self._set_terminal(job_id, JobStatus.FAILED, error=error)
        return self.get_status(job_id)

    def clear(self) -> None:
        with self._lock:
            self._jobs.clear()

    def shutdown(self, wait: bool = False, cancel_futures: bool = False) -> None:
        try:
            self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)
        except TypeError:
            self._executor.shutdown(wait=wait)


job_service = JobService()


def get_job_service() -> JobService:
    return job_service
