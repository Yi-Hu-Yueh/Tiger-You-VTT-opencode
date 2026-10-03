"""本機語音檔路由：上傳轉錄 + 背景 Job."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, UploadFile

from app.services.audio import SUPPORTED_AUDIO_EXTS, process_audio_file
from app.services.errors import AppError, to_http_exception
from app.services.jobs import get_job_service
from app.services.video import save_upload_file

router = APIRouter()


def _check_ext(filename: str) -> None:
    if Path(filename or "").suffix.lower() not in SUPPORTED_AUDIO_EXTS:
        raise AppError("unsupported_audio_format", Path(filename or "").suffix)


@router.post("/api/audio/transcript")
def audio_transcript(
    file: UploadFile = File(...),
    start_time: str | None = Form(default="0:0"),
    end_time: str | None = Form(default="0:10"),
) -> dict[str, Any]:
    _check_ext(file.filename or "")
    tmp_dir = Path(tempfile.mkdtemp(prefix="tigeraud_"))
    saved: str | None = None
    try:
        saved = save_upload_file(file, tmp_dir)
        is_default_end = (end_time or "").strip() == "0:10"
        resp = process_audio_file(saved, start_time, end_time, is_default_end)
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


@router.post("/api/jobs/audio")
def create_audio_job(
    file: UploadFile = File(...),
    start_time: str | None = Form(default="0:0"),
    end_time: str | None = Form(default="0:10"),
) -> dict[str, Any]:
    _check_ext(file.filename or "")
    svc = get_job_service()
    created = svc.create_job("audio")
    job_id: str = created["job_id"]
    job_dir = Path(tempfile.mkdtemp(prefix=f"tigerjob_{job_id}_"))
    saved = save_upload_file(file, job_dir)
    is_default_end = (end_time or "").strip() == "0:10"

    def _work() -> Any:
        try:
            cb = svc.callback_for(job_id)
            return process_audio_file(
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
