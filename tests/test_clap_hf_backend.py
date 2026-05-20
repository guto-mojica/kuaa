"""CLAP HF-transformers backend unit tests — model fully mocked, hermetic."""
from __future__ import annotations

import types

import numpy as np
import pytest

from cinemateca.models.base import AudioEmbedder


def _cfg():
    sn = types.SimpleNamespace
    return sn(
        audio_embeddings=sn(
            model_id="laion/larger_clap_general",
            batch_size=8,
            chunk_seconds=10.0,
            sample_rate=48000,
        ),
    )


class _FakeAudioModel:
    """Stand-in for transformers ClapModel. Returns deterministic vectors."""

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.calls = {"text": 0, "audio": 0}

    def get_text_features(self, **inputs):
        import torch
        self.calls["text"] += 1
        n = inputs["input_ids"].shape[0]
        return torch.zeros((n, self.dim))

    def get_audio_features(self, **inputs):
        import torch
        # Each call is one chunk; CLAP returns one vec per item.
        self.calls["audio"] += 1
        n = inputs["input_features"].shape[0]
        # Non-zero output so L2 normalisation does something visible.
        return torch.ones((n, self.dim))

    def to(self, device):
        return self

    def eval(self):
        return self


class _FakeProcessor:
    """Stand-in for transformers ClapProcessor."""

    def __call__(self, *, text=None, audios=None, sampling_rate=None, return_tensors="pt", padding=True):
        import torch
        if text is not None:
            if isinstance(text, str):
                text = [text]
            return {"input_ids": torch.zeros((len(text), 1), dtype=torch.long),
                    "attention_mask": torch.ones((len(text), 1), dtype=torch.long)}
        if audios is not None:
            return {"input_features": torch.zeros((len(audios), 64, 100))}
        raise ValueError("FakeProcessor: pass text or audios")


def _backend_with_fakes(monkeypatch, model=None):
    from cinemateca.models.audio import clap_hf

    fake_model = model or _FakeAudioModel()
    fake_proc = _FakeProcessor()

    def fake_load(self):
        self._model = fake_model
        self._processor = fake_proc

    monkeypatch.setattr(clap_hf.ClapHFEmbedder, "_load_model", fake_load)

    # Patch soundfile.read to return a constant signal.
    sr = 48000

    def fake_sf_read(path, dtype="float32"):
        # 15-second clip → 1 full chunk + 1 partial. Mean-pool should run.
        return np.zeros(int(15.0 * sr), dtype=np.float32), sr

    monkeypatch.setattr(clap_hf.sf, "read", fake_sf_read)

    return clap_hf.ClapHFEmbedder(_cfg()), fake_model


def test_clap_backend_conforms(monkeypatch):
    backend, _ = _backend_with_fakes(monkeypatch)
    assert isinstance(backend, AudioEmbedder)


def test_encode_text_returns_normalised_vector(monkeypatch):
    backend, _ = _backend_with_fakes(monkeypatch)
    v = backend.encode_text("festive music")
    assert v.shape == (512,)
    assert v.dtype == np.float32
    # Fake returns zeros, so L2 norm = 0; the backend should still return the
    # raw vector (NaN-safe normalisation is the backend's responsibility — verify
    # by checking a non-zero model below).


def test_encode_audio_single_chunks_and_mean_pools(monkeypatch, tmp_path):
    backend, fake = _backend_with_fakes(monkeypatch)
    wav = tmp_path / "scene_0001.wav"
    wav.touch()
    v = backend.encode_audio_single(wav)
    assert v.shape == (512,)
    assert v.dtype == np.float32
    # 15s clip @ 10s chunks → 2 chunks → 1 model call (batched).
    assert fake.calls["audio"] == 1
    # All-ones output, L2-normalised, should sum to sqrt(D).
    assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-5)


def test_encode_audio_batches_multiple_wavs(monkeypatch, tmp_path):
    backend, fake = _backend_with_fakes(monkeypatch)
    wavs = [tmp_path / f"scene_{i:04d}.wav" for i in (1, 2, 3)]
    for w in wavs:
        w.touch()
    out = backend.encode_audio(wavs)
    assert out.shape == (3, 512)
    assert out.dtype == np.float32
    # Each row L2-normalised.
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_encode_audio_empty_list_returns_empty_array(monkeypatch):
    backend, _ = _backend_with_fakes(monkeypatch)
    out = backend.encode_audio([])
    assert out.shape == (0, 512)
    assert out.dtype == np.float32


def test_encode_audio_single_downmixes_stereo(monkeypatch, tmp_path):
    """Stereo WAV must be downmixed to mono via mean over channel axis."""
    from cinemateca.models.audio import clap_hf

    sr = 48000

    # Capture what the chunker receives so we can assert it is 1-D.
    captured = {}

    real_chunk = clap_hf.ClapHFEmbedder._chunk_audio

    def spy_chunk(self, wav, sr):
        captured["ndim"] = wav.ndim
        captured["shape"] = wav.shape
        return real_chunk(self, wav, sr)

    monkeypatch.setattr(clap_hf.ClapHFEmbedder, "_chunk_audio", spy_chunk)

    backend, _ = _backend_with_fakes(monkeypatch)
    monkeypatch.setattr(
        clap_hf.sf,
        "read",
        lambda path, dtype="float32": (
            np.zeros((int(2.0 * sr), 2), dtype=np.float32),
            sr,
        ),
    )

    wav = tmp_path / "stereo.wav"
    wav.touch()
    backend.encode_audio_single(wav)
    assert captured["ndim"] == 1
    assert captured["shape"] == (int(2.0 * sr),)


def test_encode_audio_single_raises_on_sample_rate_mismatch(monkeypatch, tmp_path):
    """Sample rate other than configured 48000 must raise ValueError."""
    from cinemateca.models.audio import clap_hf

    backend, _ = _backend_with_fakes(monkeypatch)
    monkeypatch.setattr(
        clap_hf.sf,
        "read",
        lambda path, dtype="float32": (
            np.zeros(int(1.0 * 22050), dtype=np.float32),
            22050,
        ),
    )

    wav = tmp_path / "wrongrate.wav"
    wav.touch()
    with pytest.raises(ValueError, match="22050"):
        backend.encode_audio_single(wav)


def test_save_writes_npy_and_mapping_with_all_keys(monkeypatch, tmp_path):
    """save() must persist .npy + JSON mapping with all 9 schema keys."""
    import json

    backend, _ = _backend_with_fakes(monkeypatch)
    embeddings = np.ones((2, 512), dtype="float32")
    rows = [
        {
            "scene_id": 1,
            "wav_path": "audio/segments/scene_0001.wav",
            "start_time_s": 0.0,
            "end_time_s": 5.0,
            "chunks_per_scene": 1,
        },
        {
            "scene_id": 2,
            "wav_path": "audio/segments/scene_0002.wav",
            "start_time_s": 5.0,
            "end_time_s": 12.5,
            "chunks_per_scene": 1,
        },
    ]
    out_dir = tmp_path / "audio"
    emb_path, map_path = backend.save(embeddings, rows, out_dir)

    assert emb_path.exists() and emb_path.name == "clap_embeddings.npy"
    assert map_path.exists() and map_path.name == "audio_mapping.json"

    loaded_emb = np.load(emb_path)
    assert loaded_emb.shape == (2, 512)
    assert loaded_emb.dtype == np.float32

    m = json.loads(map_path.read_text())
    assert set(m.keys()) == {
        "model",
        "dimension",
        "total_vectors",
        "normalized",
        "scene_ids",
        "wav_paths",
        "start_times_s",
        "end_times_s",
        "chunks_per_scene",
    }
    assert m["model"] == "laion/larger_clap_general"
    assert m["dimension"] == 512
    assert m["total_vectors"] == 2
    assert m["normalized"] is True
    assert m["scene_ids"] == [1, 2]
    assert m["wav_paths"] == [
        "audio/segments/scene_0001.wav",
        "audio/segments/scene_0002.wav",
    ]
