"""系統基礎配置.

集中讀取環境變數，避免各 service 自行散落 os.environ。
所有 subprocess（FFmpeg/FFprobe）必須使用 argument array 且 shell=False。
"""

from __future__ import annotations

import os
from pathlib import Path

try:  # pydantic-settings 存在時使用 BaseSettings
    from pydantic_settings import BaseSettings, SettingsConfigDict

    _HAS_BASE_SETTINGS = True
except Exception:  # pragma: no cover
    BaseSettings = object  # type: ignore
    SettingsConfigDict = dict  # type: ignore
    _HAS_BASE_SETTINGS = False

__all__ = ["Settings", "settings", "get_settings"]


def _env_str(name: str, default: str) -> str:
    val = os.environ.get(name, default)
    return val if val != "" else default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


class Settings(BaseSettings if _HAS_BASE_SETTINGS else object):  # type: ignore
    """系統設定；欄位名小寫對應全大寫環境變數（case-insensitive）。"""

    if _HAS_BASE_SETTINGS:
        model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    WHISPER_MODEL: str = "large-v3"
    WHISPER_DEVICE: str = "auto"
    WHISPER_COMPUTE_TYPE: str = "auto"
    WHISPER_CUDA_RUNTIME_DIR: str | None = None
    SYSTEM_AUDIO_CHUNK_SECONDS: float = 10.0

    def __init__(self, **kwargs: object) -> None:
        if _HAS_BASE_SETTINGS:
            super().__init__(**kwargs)  # type: ignore
        else:
            self.WHISPER_MODEL = _env_str("WHISPER_MODEL", "large-v3")
            self.WHISPER_DEVICE = _env_str("WHISPER_DEVICE", "auto")
            self.WHISPER_COMPUTE_TYPE = _env_str("WHISPER_COMPUTE_TYPE", "auto")
            cuda_dir = os.environ.get("WHISPER_CUDA_RUNTIME_DIR", "").strip()
            self.WHISPER_CUDA_RUNTIME_DIR = cuda_dir or None
            self.SYSTEM_AUDIO_CHUNK_SECONDS = _env_float("SYSTEM_AUDIO_CHUNK_SECONDS", 10.0)

    @property
    def app_dir(self) -> Path:
        return Path(__file__).resolve().parent

    @property
    def project_root(self) -> Path:
        # config.py 位於 <root>/app/config.py；回推專案根目錄
        return Path(__file__).resolve().parent.parent


settings = Settings()


def get_settings() -> Settings:
    return settings
