"""Controlled 錯誤類型.

所有可預期失敗皆以 AppError(code) 拋出，code 即 client-safe 錯誤訊息主體，
可被 FastAPI HTTPException / Exception handler 捕獲回傳，不洩漏內部路徑。
"""

from __future__ import annotations

from typing import Any

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse

    _HAS_FASTAPI = True
except Exception:  # pragma: no cover
    FastAPI = Any  # type: ignore
    HTTPException = Exception  # type: ignore
    Request = Any  # type: ignore
    JSONResponse = None  # type: ignore
    _HAS_FASTAPI = False

__all__ = [
    "AppError",
    "raise_app_error",
    "to_http_exception",
    "register_exception_handlers",
    "ERROR_STATUS_MAP",
]

ERROR_STATUS_MAP: dict[str, int] = {
    "subtitle_track_not_found": 404,
    "no_subtitles_available": 404,
    "no_audio_available": 400,
    "audio_download_failed": 502,
    "whisper_model_load_failed": 500,
    "whisper_transcription_failed": 500,
    "whisper_empty_transcript": 500,
    "unsupported_video_format": 400,
    "invalid_video_file": 400,
    "media_probe_failed": 400,
    "embedded_subtitle_extraction_failed": 500,
    "no_transcript_source_available": 400,
    "upload_too_large": 413,
    "upload_storage_failed": 500,
    "invalid_time_range": 400,
    "job_not_found": 404,
    "unsupported_audio_format": 400,
    "invalid_audio_file": 400,
    "system_audio_unsupported": 400,
    "system_audio_device_not_found": 404,
    "system_audio_device_invalid": 400,
    "system_audio_capture_failed": 500,
    "microphone_unsupported": 400,
    "microphone_device_not_found": 404,
    "microphone_device_invalid": 400,
    "microphone_capture_failed": 500,
    "range_clip_failed": 500,
}


class AppError(Exception):
    """Controlled 業務錯誤；message 必須保持 client-safe."""

    def __init__(self, code: str, detail: str = "", status_code: int | None = None) -> None:
        self.code = code
        self.detail = detail or ""
        self.status_code = status_code if status_code is not None else ERROR_STATUS_MAP.get(code, 400)
        super().__init__(code if not self.detail else f"{code}: {self.detail}")


def raise_app_error(code: str, detail: str = "") -> None:
    raise AppError(code, detail)


def to_http_exception(exc: AppError):
    if _HAS_FASTAPI:
        from fastapi import HTTPException as _HTTP

        return _HTTP(status_code=exc.status_code, detail=exc.code if not exc.detail else f"{exc.code}: {exc.detail}")
    return exc


def register_exception_handlers(app: Any) -> None:
    if not _HAS_FASTAPI:
        return

    async def _app_error_handler(request: Request, exc: AppError):  # type: ignore
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.code, "message": exc.detail or exc.code},
        )

    async def _value_error_handler(request: Request, exc: ValueError):  # type: ignore
        code = str(exc) or "invalid_time_range"
        if code not in ERROR_STATUS_MAP:
            code = "invalid_time_range"
        return JSONResponse(status_code=ERROR_STATUS_MAP[code], content={"error": code, "message": code})

    async def _key_error_handler(request: Request, exc: KeyError):  # type: ignore
        msg = str(exc).strip("'\"")
        if msg.startswith("job_not_found"):
            return JSONResponse(status_code=404, content={"error": "job_not_found", "message": msg})
        return JSONResponse(status_code=404, content={"error": "job_not_found", "message": msg})

    app.add_exception_handler(AppError, _app_error_handler)  # type: ignore
    app.add_exception_handler(ValueError, _value_error_handler)  # type: ignore
    app.add_exception_handler(KeyError, _key_error_handler)  # type: ignore
