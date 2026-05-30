"""
cinemateca.models.transcriber.faster_whisper_hf
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
faster-whisper transcriber (CTranslate2-accelerated Whisper).

Default model: ``Systran/faster-whisper-medium`` (~1.5 GB, multilingual,
auto-detect language per scene). VAD filter on by default so archival
silence does not get hallucinated into filler text. Lazy load: the first
call to ``transcribe()`` triggers ``_load_model()``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from cinemateca.models.base import TranscriptionResult

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_ID = "Systran/faster-whisper-medium"
_DEFAULT_COMPUTE_TYPE = "auto"
_DEFAULT_BEAM_SIZE = 5
_DEFAULT_VAD_FILTER = True
_DEFAULT_VAD_MIN_SILENCE_MS = 500


def _resolve_compute_type(compute_type: str, device: str) -> str:
    """Resolve ``"auto"`` to ``float16`` on CUDA, ``int8`` on CPU."""
    if compute_type != "auto":
        return compute_type
    return "float16" if device == "cuda" else "int8"


def _load_model(cfg: Any, device: str | None):
    """Module-scoped loader so tests can monkeypatch around the network."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "Whisper transcriber requires faster-whisper. " "Install with: uv sync --extra full"
        ) from exc

    tr = getattr(cfg, "transcriber", None) if cfg else None
    model_id = str(getattr(tr, "model_id", _DEFAULT_MODEL_ID))
    compute_type_cfg = str(getattr(tr, "compute_type", _DEFAULT_COMPUTE_TYPE))

    if device is None:
        from cinemateca.device import get_device

        device = str(get_device("auto"))

    compute_type = _resolve_compute_type(compute_type_cfg, device)
    logger.info(
        "Loading faster-whisper: %s (device=%s, compute_type=%s) — first run downloads ~1.5 GB",
        model_id,
        device,
        compute_type,
    )
    return WhisperModel(model_id, device=device, compute_type=compute_type)


class FasterWhisperTranscriber:
    """faster-whisper backend implementing the ``Transcriber`` Protocol."""

    def __init__(self, cfg: Any = None, device: str | None = None) -> None:
        self._cfg = cfg
        self._device = device
        self._model: Any = None  # lazy

        tr = getattr(cfg, "transcriber", None) if cfg else None
        self._language = getattr(tr, "language", None)
        self._beam_size = int(getattr(tr, "beam_size", _DEFAULT_BEAM_SIZE))
        self._vad_filter = bool(getattr(tr, "vad_filter", _DEFAULT_VAD_FILTER))
        self._vad_min_silence_ms = int(
            getattr(tr, "vad_min_silence_duration_ms", _DEFAULT_VAD_MIN_SILENCE_MS)
        )

    def _ensure_model(self) -> None:
        if self._model is None:
            self._model = _load_model(self._cfg, self._device)

    def transcribe(self, wav_path: str | Path) -> TranscriptionResult:
        """Return a :data:`TranscriptionResult` dict; returns empty result on silent input."""
        self._ensure_model()
        segments_iter, info = self._model.transcribe(
            str(wav_path),
            language=self._language,
            beam_size=self._beam_size,
            vad_filter=self._vad_filter,
            vad_parameters={"min_silence_duration_ms": self._vad_min_silence_ms},
        )
        segments: list[dict] = []
        for seg in segments_iter:
            segments.append(
                {
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "text": str(seg.text).strip(),
                }
            )
        if not segments:
            return {
                "text": "",
                "language": None,
                "language_probability": 0.0,
                "segments": [],
            }
        text = " ".join(s["text"] for s in segments).strip()
        return {
            "text": text,
            "language": getattr(info, "language", None),
            "language_probability": float(getattr(info, "language_probability", 0.0) or 0.0),
            "segments": segments,
        }
