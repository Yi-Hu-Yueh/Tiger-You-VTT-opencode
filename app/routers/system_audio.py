"""Windows 系統音訊路由：裝置列舉 + Live Job."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.schemas.system_audio import SystemAudioJobRequest
from app.services.errors import AppError, to_http_exception
from app.services.jobs import get_job_service
from app.services.system_audio import list_loopback_devices, run_live_capture

router = APIRouter()


@router.get("/api/system-audio/devices")
def system_audio_devices() -> dict[str, Any]:
    try:
        devices = list_loopback_devices()
    except AppError as exc:
        raise to_http_exception(exc)
    return {"devices": devices}


@router.post("/api/jobs/system-audio")
def create_system_audio_job(body: SystemAudioJobRequest) -> dict[str, Any]:
    svc = get_job_service()
    created = svc.create_job("system_audio")
    job_id: str = created["job_id"]

    def _work() -> Any:
        return run_live_capture(job_id, body.device_id, svc)

    svc.submit(job_id, _work)
    return {"job_id": job_id}
