"""麥克風路由：裝置列舉 + Live Job."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.schemas.microphone import MicrophoneJobRequest
from app.services.errors import AppError, to_http_exception
from app.services.jobs import get_job_service
from app.services.microphone import list_microphone_devices, run_microphone_capture

router = APIRouter()


@router.get("/api/microphone/devices")
def microphone_devices() -> dict[str, Any]:
    try:
        devices = list_microphone_devices()
    except AppError as exc:
        raise to_http_exception(exc)
    return {"devices": devices}


@router.post("/api/jobs/microphone")
def create_microphone_job(body: MicrophoneJobRequest) -> dict[str, Any]:
    svc = get_job_service()
    created = svc.create_job("microphone")
    job_id: str = created["job_id"]

    def _work() -> Any:
        return run_microphone_capture(job_id, body.device_id, svc)

    svc.submit(job_id, _work)
    return {"job_id": job_id}
