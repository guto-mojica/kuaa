"""Transcriber Protocol conformance and empty-result contract."""

from __future__ import annotations

from pathlib import Path

from cinemateca.models.base import Transcriber


class _StubTranscriber:
    def transcribe(self, wav_path):
        return {
            "text": "hello world",
            "language": "en",
            "language_probability": 0.95,
            "segments": [{"start": 0.0, "end": 1.0, "text": "hello world"}],
        }


def test_stub_satisfies_protocol():
    t = _StubTranscriber()
    assert isinstance(t, Transcriber)


def test_protocol_accepts_str_or_path():
    t = _StubTranscriber()
    t.transcribe("/tmp/example.wav")
    t.transcribe(Path("/tmp/example.wav"))


class _EmptyResultTranscriber:
    def transcribe(self, wav_path):
        return {"text": "", "language": None, "language_probability": 0.0, "segments": []}


def test_empty_result_shape_satisfies_protocol():
    t = _EmptyResultTranscriber()
    assert isinstance(t, Transcriber)
    out = t.transcribe("/tmp/silent.wav")
    assert out["text"] == ""
    assert out["language"] is None
    assert out["segments"] == []
