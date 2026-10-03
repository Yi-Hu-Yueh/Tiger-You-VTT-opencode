"""背景 Job 查詢與 Cooperative STOP."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.schemas.jobs import JobStatusResponse
from app.services.errors import AppError, to_http_exception
from app.services.jobs import get_job_service

router = APIRouter()


@router.get("/api/jobs/{job_id}")
def get_job_status(job_id: str) -> dict[str, Any]:
    svc = get_job_service()
    try:
        status = svc.get_status(job_id)
    except KeyError:
        raise to_http_exception(AppError("job_not_found", str(job_id)))
    return JobStatusResponse(**status).model_dump()


@router.post("/api/jobs/{job_id}/stop")
def stop_job(job_id: str) -> dict[str, Any]:
    svc = get_job_service()
    try:
        status = svc.stop_job(job_id)
    except KeyError:
        raise to_http_exception(AppError("job_not_found", str(job_id)))
    return JobStatusResponse(**status).model_dump()
