"""FasterWhisperTranscriber tests — model load is monkeypatched."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cinemateca.models.base import Transcriber


def test_protocol_conformance():
    from cinemateca.models.transcriber.faster_whisper_hf import FasterWhisperTranscriber

    t = FasterWhisperTranscriber.__new__(FasterWhisperTranscriber)
    assert isinstance(t, Transcriber)


def test_transcribe_returns_full_shape(monkeypatch):
    from cinemateca.models.transcriber import faster_whisper_hf as mod

    captured: dict = {}

    class _FakeSegment:
        def __init__(self, start, end, text):
            self.start, self.end, self.text = start, end, text

    class _FakeInfo:
        language = "pt"
        language_probability = 0.93

    class _FakeModel:
        def transcribe(self, wav_path, **kwargs):
            captured.update({"wav_path": wav_path, "kwargs": kwargs})
            return (
                iter(
                    [
                        _FakeSegment(0.0, 2.3, "Aqui vai"),
                        _FakeSegment(2.3, 4.1, "o exército"),
                    ]
                ),
                _FakeInfo(),
            )

    monkeypatch.setattr(mod, "_load_model", lambda cfg, device: _FakeModel())
    cfg = SimpleNamespace(
        transcriber=SimpleNamespace(
            model_id="Systran/faster-whisper-medium",
            compute_type="auto",
            language=None,
            beam_size=5,
            vad_filter=True,
            vad_min_silence_duration_ms=500,
        )
    )
    t = mod.FasterWhisperTranscriber(cfg, device="cpu")
    out = t.transcribe("/tmp/example.wav")

    assert captured["kwargs"]["language"] is None
    assert captured["kwargs"]["beam_size"] == 5
    assert captured["kwargs"]["vad_filter"] is True
    assert captured["kwargs"]["vad_parameters"]["min_silence_duration_ms"] == 500

    assert out["text"] == "Aqui vai o exército"
    assert out["language"] == "pt"
    assert out["language_probability"] == pytest.approx(0.93)
    assert len(out["segments"]) == 2
    assert out["segments"][0] == {"start": 0.0, "end": 2.3, "text": "Aqui vai"}


def test_empty_segments_returns_empty_result(monkeypatch):
    from cinemateca.models.transcriber import faster_whisper_hf as mod

    class _FakeInfo:
        language = None
        language_probability = 0.0

    class _FakeModel:
        def transcribe(self, wav_path, **kwargs):
            return iter([]), _FakeInfo()

    monkeypatch.setattr(mod, "_load_model", lambda cfg, device: _FakeModel())
    cfg = SimpleNamespace(
        transcriber=SimpleNamespace(
            model_id="Systran/faster-whisper-medium",
            compute_type="auto",
            language=None,
            beam_size=5,
            vad_filter=True,
            vad_min_silence_duration_ms=500,
        )
    )
    t = mod.FasterWhisperTranscriber(cfg, device="cpu")
    out = t.transcribe("/tmp/silent.wav")
    assert out == {"text": "", "language": None, "language_probability": 0.0, "segments": []}


def test_lazy_load(monkeypatch):
    from cinemateca.models.transcriber import faster_whisper_hf as mod

    loaded = {"flag": False}

    def _spy_load(cfg, device):
        loaded["flag"] = True
        return object()

    monkeypatch.setattr(mod, "_load_model", _spy_load)
    cfg = SimpleNamespace(
        transcriber=SimpleNamespace(
            model_id="x",
            compute_type="auto",
            language=None,
            beam_size=5,
            vad_filter=True,
            vad_min_silence_duration_ms=500,
        )
    )
    mod.FasterWhisperTranscriber(cfg, device="cpu")
    assert loaded["flag"] is False, "Construction must not load the model"


def test_registry_returns_faster_whisper(monkeypatch):
    from cinemateca.models.registry import get_transcriber
    from cinemateca.models.transcriber import faster_whisper_hf as mod

    monkeypatch.setattr(mod, "_load_model", lambda cfg, device: object())

    cfg = SimpleNamespace(
        models=SimpleNamespace(transcriber="faster_whisper_hf"),
        transcriber=SimpleNamespace(
            model_id="Systran/faster-whisper-medium",
            compute_type="auto",
            language=None,
            beam_size=5,
            vad_filter=True,
            vad_min_silence_duration_ms=500,
        ),
    )
    t = get_transcriber(cfg, device="cpu")
    assert isinstance(t, Transcriber)


def test_registry_rejects_unknown(monkeypatch):
    from cinemateca.models.registry import get_transcriber

    cfg = SimpleNamespace(models=SimpleNamespace(transcriber="nonexistent_backend"))
    with pytest.raises(ValueError, match="Unknown transcriber"):
        get_transcriber(cfg)
