"""FastAPI 入口：註冊 routers、錯誤處理、健康檢查."""

from __future__ import annotations

from fastapi import FastAPI

from app.routers import audio, jobs, system_audio, ui, video, youtube
from app.services.errors import register_exception_handlers

app = FastAPI(title="Tiger-You-VTT", version="0.1.0")

register_exception_handlers(app)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(youtube.router)
app.include_router(video.router)
app.include_router(audio.router)
app.include_router(system_audio.router)
app.include_router(jobs.router)
app.include_router(ui.router)
