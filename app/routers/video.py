"""本機影片路由：上傳 + 同步 / 背景 Job."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, UploadFile

from app.services.errors import AppError, to_http_exception
from app.services.jobs import get_job_service
from app.services.video import SUPPORTED_VIDEO_EXTS, process_video_file, save_upload_file

router = APIRouter()

MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024


def _check_ext(filename: str) -> None:
    if Path(filename or "").suffix.lower() not in SUPPORTED_VIDEO_EXTS:
        raise AppError("unsupported_video_format", Path(filename or "").suffix)


@router.post("/api/video/subtitle")
def video_subtitle(
    file: UploadFile = File(...),
    start_time: str | None = Form(default="0:0"),
    end_time: str | None = Form(default="0:10"),
) -> dict[str, Any]:
    _check_ext(file.filename or "")
    tmp_dir = Path(tempfile.mkdtemp(prefix="tigervid_"))
    saved: str | None = None
    try:
        saved = save_upload_file(file, tmp_dir)
        if Path(saved).stat().st_size > MAX_UPLOAD_BYTES:
            raise AppError("upload_too_large")
        is_default_end = (end_time or "").strip() == "0:10"
        resp = process_video_file(saved, start_time, end_time, is_default_end)
        return resp.model_dump()
    except AppError as exc:
        raise to_http_exception(exc)
    except ValueError as exc:
        raise to_http_exception(AppError(str(exc) or "invalid_time_range"))
    finally:
        try:
            if saved and Path(saved).is_file():
                os.remove(saved)
            tmp_dir.rmdir()
        except Exception:
            pass


@router.post("/api/jobs/video")
def create_video_job(
    file: UploadFile = File(...),
    start_time: str | None = Form(default="0:0"),
    end_time: str | None = Form(default="0:10"),
) -> dict[str, Any]:
    _check_ext(file.filename or "")
    svc = get_job_service()
    created = svc.create_job("video")
    job_id: str = created["job_id"]
    # job-owned 暫存：terminal 前不刪除
    job_dir = Path(tempfile.mkdtemp(prefix=f"tigerjob_{job_id}_"))
    saved = save_upload_file(file, job_dir)
    is_default_end = (end_time or "").strip() == "0:10"

    def _work() -> Any:
        try:
            cb = svc.callback_for(job_id)
            return process_video_file(
                saved, start_time, end_time, is_default_end,
                job_id=job_id,
                stop_checker=lambda: svc.stop_checker(job_id),
                progress_callback=cb,
            )
        finally:
            try:
                if Path(saved).is_file():
                    os.remove(saved)
                job_dir.rmdir()
            except Exception:
                pass

    svc.submit(job_id, _work)
    return {"job_id": job_id}
