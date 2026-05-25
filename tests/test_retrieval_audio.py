"""Audio-only retrieval service tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cinemateca.search.audio import (
    AudioIndex,
    load_audio_index,
)


def _write_fake_clap_index(film_dir: Path, n: int = 4, dim: int = 512) -> None:
    """Write a deterministic fake (embeddings, mapping) pair under
    ``film_dir/audio/`` matching the CLAP writer's on-disk format."""
    audio_dir = film_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((n, dim)).astype("float32")
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    np.save(audio_dir / "clap_embeddings.npy", emb)
    mapping = [{"scene_id": i, "wav_path": f"audio/scene_{i:04d}.wav"} for i in range(n)]
    (audio_dir / "audio_mapping.json").write_text(json.dumps(mapping))


def test_load_audio_index_returns_l2_normalised_matrix(tmp_path: Path) -> None:
    _write_fake_clap_index(tmp_path, n=4, dim=512)
    idx = load_audio_index(tmp_path / "audio")
    assert isinstance(idx, AudioIndex)
    assert idx.embeddings.shape == (4, 512)
    assert idx.embeddings.dtype == np.float32
    norms = np.linalg.norm(idx.embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)
    assert [m["scene_id"] for m in idx.mapping] == [0, 1, 2, 3]


def test_load_audio_index_missing_files_returns_none(tmp_path: Path) -> None:
    assert load_audio_index(tmp_path / "audio") is None
