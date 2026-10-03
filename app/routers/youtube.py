"""YouTube 路由：同步字幕 + 背景 Job."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.schemas.youtube import YoutubeInfoRequest, YoutubeJobRequest, YoutubeSubtitleRequest
from app.services.errors import AppError, to_http_exception
from app.services.jobs import get_job_service
from app.services.youtube import fetch_video_info, process_youtube_url

router = APIRouter()


@router.post("/api/youtube/info")
def youtube_info(body: YoutubeInfoRequest) -> dict[str, Any]:
    try:
        info = fetch_video_info(body.url)
    except AppError as exc:
        raise to_http_exception(exc)
    return {
        "title": info.get("title"),
        "duration": info.get("duration"),
        "subtitles": sorted((info.get("subtitles") or {}).keys()),
        "automatic_captions": sorted((info.get("automatic_captions") or {}).keys()),
    }


@router.post("/api/youtube/subtitle")
def youtube_subtitle(body: YoutubeSubtitleRequest) -> dict[str, Any]:
    try:
        is_default_end = (body.end_time or "").strip() == "0:10"
        resp = process_youtube_url(body.url, body.start_time, body.end_time, is_default_end)
    except AppError as exc:
        raise to_http_exception(exc)
    except ValueError as exc:
        raise to_http_exception(AppError(str(exc) or "invalid_time_range"))
    return resp.model_dump()


@router.post("/api/jobs/youtube")
def create_youtube_job(body: YoutubeJobRequest) -> dict[str, Any]:
    svc = get_job_service()
    created = svc.create_job("youtube")
    job_id: str = created["job_id"]
    is_default_end = (body.end_time or "").strip() == "0:10"

    def _work() -> Any:
        cb = svc.callback_for(job_id)
        return process_youtube_url(
            body.url, body.start_time, body.end_time, is_default_end,
            job_id=job_id,
            stop_checker=lambda: svc.stop_checker(job_id),
            progress_callback=cb,
        )

    svc.submit(job_id, _work)
    return {"job_id": job_id}
