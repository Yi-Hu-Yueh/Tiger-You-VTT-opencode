"""本機 Web UI：GET / 回傳單檔 index.html."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter()

_INDEX = Path(__file__).resolve().parent.parent / "static" / "index.html"


@router.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(str(_INDEX), media_type="text/html; charset=utf-8")
