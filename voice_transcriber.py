"""Offline voice-note transcription helper for EchoText.

Supports two local speech-to-text backends when installed on the user's PC:
1. ``faster-whisper`` Python package (if installed in virtualenv)
2. ``whisper.cpp`` CLI binary (``WHISPER_BINARY_PATH`` + ``WHISPER_MODEL_PATH``)

If neither speech model is installed or transcription fails, returns ``None`` so
``telegram_listener.py`` falls back to the polite canned voice-note message instead
of posting an empty string to the LLM.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess  # nosec B404
from typing import Any

from config import WHISPER_BINARY_PATH, WHISPER_MODEL_PATH, setup_logging

logger = setup_logging("echotext.voice")
_FASTER_WHISPER_MODEL: Any = None


def transcribe_voice_note(
    audio_path: Path,
    model_path: Path | None = None,
    whisper_bin: str = WHISPER_BINARY_PATH,
    timeout_seconds: int = 30,
) -> str | None:
    """Transcribe a downloaded voice note locally if a speech model is available on the PC."""
    if not audio_path.exists() or audio_path.stat().st_size == 0:
        return None

    # 1. Try faster-whisper if installed
    global _FASTER_WHISPER_MODEL
    try:
        from faster_whisper import WhisperModel  # type: ignore[import-not-found]

        if _FASTER_WHISPER_MODEL is None:
            model_id = (
                str(model_path) if (model_path and model_path.exists()) else "base"
            )
            _FASTER_WHISPER_MODEL = WhisperModel(
                model_id, device="cpu", compute_type="int8"
            )
        segments, _ = _FASTER_WHISPER_MODEL.transcribe(str(audio_path), beam_size=2)
        text = " ".join(seg.text.strip() for seg in segments if seg.text.strip()).strip()
        if text:
            return text
    except ImportError as exc:
        logger.debug("faster-whisper not installed: %s", exc)
    except Exception:
        logger.exception("faster-whisper transcription failed for %s", audio_path)

    # 2. Try whisper.cpp CLI binary if present on PATH and model file exists
    target_model = WHISPER_MODEL_PATH if model_path is None else model_path
    resolved_bin = shutil.which(whisper_bin)
    if resolved_bin and target_model.exists():
        try:
            proc = subprocess.run(  # nosec B603  # noqa: S603
                [
                    resolved_bin,
                    "-m",
                    str(target_model),
                    "-f",
                    str(audio_path),
                    "--no-timestamps",
                    "-nt",
                ],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                return proc.stdout.strip()
        except Exception:
            logger.exception("whisper.cpp CLI transcription failed for %s", audio_path)

    return None
