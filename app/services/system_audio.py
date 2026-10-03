"""Windows WASAPI Loopback 系統音訊即時轉錄."""

from __future__ import annotations

import logging
import os
import tempfile
import wave
from pathlib import Path
from typing import Any, Callable

from app.schemas.transcript import TranscriptResponse, TranscriptSegment
from app.services.errors import AppError
from app.services.transcript import to_srt, to_txt, to_vtt

logger = logging.getLogger(__name__)

__all__ = [
    "list_loopback_devices",
    "get_loopback_device",
    "run_live_capture",
    "capture_chunk_to_wav",
]


def _import_pyaudiowpatch() -> Any:
    try:
        import pyaudiowpatch as pyaudio  # type: ignore

        return pyaudio
    except Exception as exc:
        raise AppError("system_audio_unsupported", "PyAudioWPatch/WASAPI unavailable") from exc


def list_loopback_devices() -> list[dict[str, Any]]:
    """僅列 WASAPI loopback output-capture 裝置；絕不列麥克風."""
    pyaudio = _import_pyaudiowpatch()
    try:
        pa = pyaudio.PyAudio()
    except Exception as exc:
        raise AppError("system_audio_unsupported", str(exc)[:200]) from exc
    out: list[dict[str, Any]] = []
    try:
        count = pa.get_device_count()
        for i in range(count):
            try:
                info = pa.get_device_info_by_index(i)
            except Exception:
                continue
            try:
                if int(info.get("maxInputChannels", 0)) <= 0:
                    continue
                host_idx = info.get("hostApi")
                host_name = ""
                try:
                    host_name = str(pa.get_host_api_info_by_index(host_idx).get("name", ""))
                except Exception:
                    host_name = str(info.get("hostApiName", ""))
                if "wasapi" not in host_name.lower():
                    continue
                # loopback 判定：有旗標以旗標為準，否則以名稱為準（排除麥克風）
                if "isLoopbackDevice" in info:
                    if not info.get("isLoopbackDevice"):
                        continue
                else:
                    if "loopback" not in str(info.get("name", "")).lower():
                        continue
                out.append(
                    {
                        "id": i,
                        "index": i,
                        "name": str(info.get("name", "")),
                        "sample_rate": int(float(info.get("defaultSampleRate", 44100))),
                        "channels": int(info.get("maxInputChannels", 2)),
                        "host_api": host_name,
                    }
                )
            except Exception:
                continue
    finally:
        try:
            pa.terminate()
        except Exception:
            pass
    return out


def get_loopback_device(device_id: int | None) -> dict[str, Any]:
    devices = list_loopback_devices()
    if not devices:
        raise AppError("system_audio_device_not_found", "no loopback output device")
    if device_id is None:
        return devices[0]
    for d in devices:
        if int(d["id"]) == int(device_id):
            return d
    raise AppError("system_audio_device_invalid", f"device_id {device_id} invalid")


def capture_chunk_to_wav(pcm_bytes: bytes, sample_rate: int, channels: int, out_path: str | Path) -> str:
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(int(channels))
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm_bytes)
    return str(out_path)


def _chunk_seconds() -> float:
    try:
        from app.config import get_settings

        return float(get_settings().SYSTEM_AUDIO_CHUNK_SECONDS or 10.0)
    except Exception:
        try:
            return float(os.environ.get("SYSTEM_AUDIO_CHUNK_SECONDS", "10.0"))
        except Exception:
            return 10.0


def run_live_capture(
    job_id: str,
    device_id: int | None,
    job_service: Any,
    whisper_manager: Any | None = None,
    chunk_seconds: float | None = None,
    max_chunks: int | None = None,
) -> TranscriptResponse:
    """open-ended live 擷取迴圈；STOP 停止後續 capture，當前 chunk 安全邊界完成."""
    device = get_loopback_device(device_id)
    seconds = float(chunk_seconds) if chunk_seconds else _chunk_seconds()
    if seconds <= 0:
        seconds = 10.0

    if whisper_manager is None:
        from app.services.whisper import get_whisper_manager

        whisper_manager = get_whisper_manager()

    pyaudio = _import_pyaudiowpatch()
    sample_rate = int(device["sample_rate"])
    channels = int(device["channels"])
    dev_index = int(device.get("index", device.get("id", 0)))

    stop_checker: Callable[[], bool] = lambda: bool(job_service.stop_checker(job_id))

    def _progress_offset(offset: float) -> Callable[[Any], Any]:
        def _cb(item: Any) -> Any:
            d = dict(item) if isinstance(item, dict) else {"start": item.get("start"), "end": item.get("end"), "text": item.get("text")}
            try:
                d["start"] = float(d.get("start", 0.0)) + offset
                d["end"] = float(d.get("end", d.get("start", 0.0))) + offset
            except Exception:
                pass
            try:
                job_service.update_progress_callback(job_id, d)
            except Exception:
                pass
            return None

        return _cb

    tmp_dir = Path(tempfile.mkdtemp(prefix="tigersys_"))
    pa = None
    stream = None
    session_offset = 0.0
    chunk_index = 0
    try:
        try:
            pa = pyaudio.PyAudio()
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=sample_rate,
                input=True,
                input_device_index=dev_index,
                frames_per_buffer=1024,
            )
        except Exception as exc:
            raise AppError("system_audio_capture_failed", str(exc)[:300]) from exc

        frames_per_chunk = int(sample_rate * seconds)
        while True:
            if stop_checker():
                break
            if max_chunks is not None and chunk_index >= max_chunks:
                break
            try:
                pcm = stream.read(frames_per_chunk, exception_on_overflow=False)
            except Exception as exc:
                if stop_checker():
                    break
                raise AppError("system_audio_capture_failed", str(exc)[:300]) from exc
            if isinstance(pcm, bytearray):
                pcm = bytes(pcm)

            wav_path = str(tmp_dir / f"chunk_{chunk_index:05d}.wav")
            try:
                capture_chunk_to_wav(pcm, sample_rate, channels, wav_path)
                # 實際時長（若 read 短少，以位元組推算）
                actual = seconds
                try:
                    actual = len(pcm) / float(sample_rate * channels * 2)
                    if actual <= 0:
                        actual = seconds
                except Exception:
                    actual = seconds
                # Live：is_live_chunk=True，0 segments 直接接受，絕不 VAD-off 重試
                raw = whisper_manager.transcribe_file(
                    wav_path,
                    vad_filter=True,
                    callback=_progress_offset(session_offset),
                    stop_checker=stop_checker,
                    is_live_chunk=True,
                )
                # transcribe 內已即時回報 offset 後 segments，此處無需重複附加
                _ = raw
            finally:
                try:
                    if Path(wav_path).is_file():
                        os.remove(wav_path)
                except Exception:
                    pass

            # session-relative timeline 推進（以要求 chunk 長度為準，維持等距）
            try:
                session_offset += float(seconds)
            except Exception:
                session_offset += 10.0
            chunk_index += 1
            if stop_checker():
                break

        status = job_service.get_status(job_id)
        segs = [TranscriptSegment(start=s.start, end=s.end, text=s.text) for s in status.get("current_segments", [])]
        return TranscriptResponse(
            language=None, type="transcribed", selection_mode="direct",
            segment_count=len(segs), duration=session_offset, segments=segs,
            vtt=to_vtt(segs), txt=to_txt(segs), srt=to_srt(segs),
        )
    finally:
        try:
            if stream is not None:
                try:
                    stream.stop_stream()
                except Exception:
                    pass
                try:
                    stream.close()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            if pa is not None:
                pa.terminate()
        except Exception:
            pass
        try:
            for cand in tmp_dir.glob("chunk_*.wav"):
                try:
                    if cand.is_file():
                        os.remove(cand)
                except Exception:
                    pass
            tmp_dir.rmdir()
        except Exception:
            pass
