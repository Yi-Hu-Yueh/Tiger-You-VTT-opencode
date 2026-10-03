"""faster-whisper 本機 ASR 核心服務.

- Lazy Loading：FastAPI startup 不載入模型，首次 ASR 才初始化 large-v3。
- Windows Process-local CUDA DLL 探索：優先 WHISPER_CUDA_RUNTIME_DIR，
  否則自動尋找目前 venv 下 venv\\Lib\\site-packages\\torch\\lib。
- GPU 自動回退：優先 cuda / int8_float32，inference 失敗則回 cpu / int8。
- VAD 策略：一般檔案 0 usable segments 僅重試一次 vad_filter=False；
  System Audio live chunk 0 segments 直接接受，絕不重試，避免 hallucination。
- Cooperative STOP：每完成一個 segment 即 callback 並檢查 stop flag。

注意：本檔案不使用 subprocess。若未來需要 FFmpeg/FFprobe，
必須使用 argument array 且 shell=False（Windows 安全）。
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Callable

try:  # 允許在未安裝 faster-whisper 的環境 import（測試以 mock 注入）
    from faster_whisper import WhisperModel as _ImportedWhisperModel  # type: ignore
except Exception:  # pragma: no cover
    _ImportedWhisperModel = None  # type: ignore

# 供測試 patch `app.services.whisper.WhisperModel` 使用（where-used）。
WhisperModel = _ImportedWhisperModel

logger = logging.getLogger(__name__)

__all__ = ["WhisperModelManager", "get_whisper_manager"]

_DEFAULT_MODEL = "large-v3"
_DEFAULT_CUDA_COMPUTE = "int8_float32"
_DEFAULT_CPU_COMPUTE = "int8"

_REQUIRED_CUDA_DLLS = ("cublas64_12.dll", "cublasLt64_12.dll")


def _resolve_model_class() -> Any:
    """同時支援 patch `faster_whisper.WhisperModel` 與 `app.services.whisper.WhisperModel`."""
    where_used = globals().get("WhisperModel")
    try:
        import faster_whisper as _fw  # type: ignore

        fresh = getattr(_fw, "WhisperModel", None)
    except Exception:
        fresh = None
    # 若 where-used 被 patch 成 Mock（與 fresh 不同），優先採用 where-used。
    if where_used is not None and fresh is not None and where_used is not fresh:
        return where_used
    if fresh is not None:
        return fresh
    if where_used is not None:
        return where_used
    raise RuntimeError("whisper_model_load_failed: faster-whisper is not installed")


class WhisperModelManager:
    """large-v3 模型生命週期管理（process 內重用成功模型）。"""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
        cuda_runtime_dir: str | None = None,
    ) -> None:
        self._model_name = (model_name or os.environ.get("WHISPER_MODEL", _DEFAULT_MODEL)).strip() or _DEFAULT_MODEL
        self._device = (device or os.environ.get("WHISPER_DEVICE", "auto")).strip().lower() or "auto"
        self._compute_type = (
            (compute_type or os.environ.get("WHISPER_COMPUTE_TYPE", "auto")).strip() or "auto"
        )
        self._cuda_runtime_dir_override = (
            (cuda_runtime_dir if cuda_runtime_dir is not None else os.environ.get("WHISPER_CUDA_RUNTIME_DIR", "")).strip()
            or None
        )
        self._model: Any | None = None
        self._working_device: str | None = None
        self._working_compute: str | None = None
        self._dll_handles: list[Any] = []
        self._cuda_dll_dir_added: str | None = None
        self._lock = threading.Lock()

    # -- properties -----------------------------------------------------
    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def working_device(self) -> str | None:
        return self._working_device

    @property
    def working_compute(self) -> str | None:
        return self._working_compute

    # -- CUDA DLL 探索 ---------------------------------------------------
    def _resolve_cuda_runtime_dir(self) -> Path | None:
        if self._cuda_runtime_dir_override:
            return Path(self._cuda_runtime_dir_override)
        candidates: list[Path] = [Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"]
        try:
            import torch  # type: ignore

            candidates.append(Path(torch.__file__).resolve().parent / "lib")
        except Exception:
            pass
        for cand in candidates:
            try:
                if cand.is_dir():
                    return cand
            except Exception:
                continue
        return None

    def _ensure_cuda_dll_directory(self) -> str | None:
        """若偵測到 cuBLAS DLL，以 process-local 方式加入搜尋路徑並保留 handle。"""
        if os.name != "nt" or not hasattr(os, "add_dll_directory"):
            return None
        runtime_dir = self._resolve_cuda_runtime_dir()
        if runtime_dir is None:
            return None
        try:
            for name in _REQUIRED_CUDA_DLLS:
                if not (runtime_dir / name).is_file():
                    return None
        except Exception:
            return None
        if self._cuda_dll_dir_added == str(runtime_dir):
            return str(runtime_dir)
        try:
            handle = os.add_dll_directory(str(runtime_dir))  # type: ignore[attr-defined]
        except Exception as exc:
            logger.debug("add_dll_directory failed: %s", exc)
            return None
        self._dll_handles.append(handle)
        self._cuda_dll_dir_added = str(runtime_dir)
        logger.info("Added CUDA runtime DLL directory: %s", runtime_dir)
        return str(runtime_dir)

    # -- 模型建立 --------------------------------------------------------
    def _candidate_devices(self) -> list[tuple[str, str]]:
        dev = (self._device or "auto").lower()
        comp_raw = (self._compute_type or "auto").strip()
        comp = comp_raw.lower()
        if dev == "cpu":
            return [("cpu", comp_raw if comp != "auto" else _DEFAULT_CPU_COMPUTE)]
        if dev == "cuda":
            return [("cuda", comp_raw if comp != "auto" else _DEFAULT_CUDA_COMPUTE)]
        # auto：優先 CUDA，失敗回 CPU
        if comp != "auto":
            return [("cuda", comp_raw), ("cpu", comp_raw)]
        return [("cuda", _DEFAULT_CUDA_COMPUTE), ("cpu", _DEFAULT_CPU_COMPUTE)]

    def _create_model(self, device: str, compute_type: str) -> Any:
        if device == "cuda":
            self._ensure_cuda_dll_directory()
        model_cls = _resolve_model_class()
        try:
            return model_cls(self._model_name, device=device, compute_type=compute_type)
        except Exception as exc:
            raise RuntimeError(f"whisper_model_load_failed: {device}/{compute_type}: {exc}") from exc

    # -- STOP / live 參數解析 ---------------------------------------------
    @staticmethod
    def _build_stop_checker(
        stop_checker: Callable[[], bool] | None,
        extra: dict[str, Any],
    ) -> Callable[[], bool]:
        if callable(stop_checker):
            return stop_checker
        for key in ("stop_event", "stop_requested", "is_stop_requested", "should_stop"):
            val = extra.get(key)
            if val is None:
                continue
            if hasattr(val, "is_set") and callable(getattr(val, "is_set")):
                return val.is_set  # type: ignore[return-value]
            if callable(val):
                return val  # type: ignore[return-value]
            flag = bool(val)
            return lambda _f=flag: _f
        return lambda: False

    @staticmethod
    def _resolve_live_flag(is_live_chunk: bool, extra: dict[str, Any]) -> tuple[bool, bool]:
        live = bool(is_live_chunk)
        for key in ("is_live", "live_mode", "live", "is_live_chunk"):
            if key in extra and bool(extra[key]):
                live = True
        allow_retry = not live
        if "allow_vad_retry" in extra:
            allow_retry = bool(extra["allow_vad_retry"])
        elif "retry_on_empty" in extra:
            allow_retry = bool(extra["retry_on_empty"])
        return live, allow_retry

    # -- 單次轉錄 ----------------------------------------------------------
    def _transcribe_once(
        self,
        model: Any,
        file_path: Path,
        vad_filter: bool,
        callback: Callable[[dict], Any] | None,
        is_stopped: Callable[[], bool],
        language: str | None,
        beam_size: int = 5,
    ) -> tuple[list[dict], Any]:
        segments_iter, info = model.transcribe(
            str(file_path),
            vad_filter=vad_filter,
            language=language,
            beam_size=beam_size,
        )
        results: list[dict] = []
        for seg in segments_iter:
            if isinstance(seg, dict):
                raw_start = seg.get("start", 0.0)
                raw_end = seg.get("end", raw_start)
                raw_text = seg.get("text", "")
            else:
                raw_start = getattr(seg, "start", 0.0)
                raw_end = getattr(seg, "end", raw_start)
                raw_text = getattr(seg, "text", "")
            try:
                start = float(raw_start)
            except Exception:
                start = 0.0
            try:
                end = float(raw_end)
            except Exception:
                end = start
            text = str(raw_text or "").strip()
            if not text:
                continue
            item = {"start": start, "end": end, "text": text}
            results.append(item)
            if callback is not None:
                try:
                    ret = callback(item)
                    if ret is True:
                        break
                except Exception:
                    pass
            try:
                if is_stopped():
                    break
            except Exception:
                pass
        return results, info

    # -- 公開 API ----------------------------------------------------------
    def transcribe_file(
        self,
        file_path: str | Path,
        vad_filter: bool = True,
        callback: Callable[[dict], Any] | None = None,
        *,
        is_live_chunk: bool = False,
        stop_checker: Callable[[], bool] | None = None,
        language: str | None = None,
        beam_size: int = 5,
        **kwargs: Any,
    ) -> list[dict]:
        """轉錄單一音訊檔，回傳 Canonical ``[{start, end, text}]``。

        - 一般檔案：0 usable segments 且非 STOP 時，僅重試一次 ``vad_filter=False``。
        - System Audio live chunk（``is_live_chunk=True`` 或別名）：0 segments
          直接接受，絕不執行 VAD-off retry。
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"audio file not found: {path}")

        live, allow_retry = self._resolve_live_flag(is_live_chunk, kwargs)
        # 相容別名：callback 可能以 keyword 傳入
        if callback is None and callable(kwargs.get("callback")):
            callback = kwargs["callback"]
        is_stopped = self._build_stop_checker(stop_checker if stop_checker is not None else kwargs.get("stop_checker"), kwargs)

        candidates = self._candidate_devices()
        last_error: Exception | None = None

        with self._lock:
            for device, compute in candidates:
                # 重用已快取的健康模型（同 device/compute 才重用）
                model = self._model if (self._model is not None and self._working_device == device and self._working_compute == compute) else None
                if model is None:
                    try:
                        model = self._create_model(device, compute)
                    except Exception as exc:
                        last_error = exc
                        continue
                try:
                    results, _info = self._transcribe_once(
                        model, path, vad_filter, callback, is_stopped, language, beam_size
                    )
                except Exception as exc:
                    # 推論期失敗：清除錯誤快取，嘗試下一個（CPU fallback）
                    last_error = RuntimeError(f"whisper_transcription_failed: {device}/{compute}: {exc}")
                    if self._model is model:
                        self._model = None
                        self._working_device = None
                        self._working_compute = None
                    continue

                # VAD retry：僅一般檔案、初始 vad_filter=True、非 STOP 才重試一次
                stopped = False
                try:
                    stopped = bool(is_stopped())
                except Exception:
                    stopped = False
                if len(results) == 0 and vad_filter and allow_retry and not live and not stopped:
                    try:
                        results, _info = self._transcribe_once(
                            model, path, False, callback, is_stopped, language, beam_size
                        )
                    except Exception as exc:
                        last_error = RuntimeError(f"whisper_transcription_failed: {device}/{compute} vad-off: {exc}")
                        if self._model is model:
                            self._model = None
                            self._working_device = None
                            self._working_compute = None
                        continue

                # 成功（含被接受的 0 segments）：快取健康模型
                self._model = model
                self._working_device = device
                self._working_compute = compute
                return results

        raise RuntimeError(f"whisper_transcription_failed: {last_error}") from last_error


_manager: WhisperModelManager | None = None
_manager_lock = threading.Lock()


def get_whisper_manager() -> WhisperModelManager:
    """取得 process 級 singleton（模型仍為 lazy 載入）。"""
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = WhisperModelManager()
        return _manager
