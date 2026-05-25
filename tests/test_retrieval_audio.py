"""Audio-only retrieval service tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

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


def _write_real_clap_index(film_dir: Path, n: int = 4, dim: int = 512) -> None:
    """Write the dict-of-parallel-arrays shape that ClapHFEmbedder.save() emits."""
    audio_dir = film_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    emb = rng.standard_normal((n, dim)).astype("float32")
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    np.save(audio_dir / "clap_embeddings.npy", emb)
    mapping = {
        "model": "laion/larger_clap_general",
        "dimension": dim,
        "total_vectors": n,
        "normalized": True,
        "scene_ids": list(range(n)),
        "wav_paths": [f"audio/scene_{i:04d}.wav" for i in range(n)],
        "start_times_s": [float(i * 5) for i in range(n)],
        "end_times_s": [float((i + 1) * 5) for i in range(n)],
    }
    (audio_dir / "audio_mapping.json").write_text(json.dumps(mapping))


def test_load_audio_index_normalises_real_dict_shape(tmp_path: Path) -> None:
    _write_real_clap_index(tmp_path, n=4, dim=512)
    idx = load_audio_index(tmp_path / "audio")
    assert idx is not None
    assert isinstance(idx.mapping, list)
    assert len(idx.mapping) == 4
    assert idx.mapping[0]["scene_id"] == 0
    assert idx.mapping[2]["scene_id"] == 2
    assert "wav_path" in idx.mapping[0]
    assert idx.mapping[1]["start_time_s"] == 5.0
    assert idx.mapping[1]["end_time_s"] == 10.0


def test_load_audio_index_rejects_unknown_mapping_shape(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    rng = np.random.default_rng(2)
    emb = rng.standard_normal((2, 8)).astype("float32")
    np.save(audio_dir / "clap_embeddings.npy", emb)
    (audio_dir / "audio_mapping.json").write_text(json.dumps({"wrong": "shape"}))
    with pytest.raises(ValueError, match="mapping shape"):
        load_audio_index(audio_dir)


def test_load_audio_index_rejects_row_count_mismatch(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    rng = np.random.default_rng(3)
    emb = rng.standard_normal((3, 8)).astype("float32")
    np.save(audio_dir / "clap_embeddings.npy", emb)
    (audio_dir / "audio_mapping.json").write_text(
        json.dumps(
            {
                "scene_ids": [10, 20],
                "wav_paths": ["a.wav", "b.wav"],
                "start_times_s": [0.0, 1.0],
                "end_times_s": [1.0, 2.0],
            }
        )
    )
    with pytest.raises(ValueError, match="row count"):
        load_audio_index(audio_dir)


def test_search_audio_returns_top_k_by_cosine(tmp_path: Path) -> None:
    _write_fake_clap_index(tmp_path, n=8, dim=4)
    idx = load_audio_index(tmp_path / "audio")
    assert idx is not None

    # Fake an embedder whose encode_text returns the first row of the
    # matrix — top-1 must then be scene_id=0 with score = 1.0.
    class _StubEmbedder:
        def encode_text(self, text: str) -> np.ndarray:
            return idx.embeddings[0].copy()

    from cinemateca.search.audio import search_audio

    hits = search_audio(idx, _StubEmbedder(), "anything", top_k=3)
    assert [h["scene_id"] for h in hits][0] == 0
    assert hits[0]["score"] == pytest.approx(1.0, abs=1e-5)
    assert len(hits) == 3
    # Scores monotonic descending
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_search_audio_top_k_clamped_to_index_size(tmp_path: Path) -> None:
    _write_fake_clap_index(tmp_path, n=3, dim=4)
    idx = load_audio_index(tmp_path / "audio")

    class _Stub:
        def encode_text(self, text: str) -> np.ndarray:
            return np.ones(4, dtype="float32") / 2.0

    from cinemateca.search.audio import search_audio

    hits = search_audio(idx, _Stub(), "x", top_k=999)
    assert len(hits) == 3


def test_load_audio_index_caches_by_stat(tmp_path: Path) -> None:
    _write_fake_clap_index(tmp_path, n=2, dim=4)
    idx1 = load_audio_index(tmp_path / "audio")
    idx2 = load_audio_index(tmp_path / "audio")
    assert idx1 is idx2, "second load on unchanged files should hit cache"
