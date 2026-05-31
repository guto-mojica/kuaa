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


# ── Typed audio verbs (C9): per-query metadata on SearchResult ────────────────


class _RowStub:
    """Embedder whose ``encode_text`` returns the index's first row."""

    def __init__(self, index: AudioIndex) -> None:
        self._index = index

    def encode_text(self, text: str) -> np.ndarray:
        return self._index.embeddings[0].copy()


def test_find_audio_returns_searchresult_with_audio_metadata(tmp_path: Path) -> None:
    """find_audio → typed SearchResult carrying the 5 C9 fields with audio
    semantics: retriever_mode='audio', fusion_used False, reranker_applied
    False, num_films_searched 1 (per-film), latency_ms timed."""
    from cinemateca.search.audio import find_audio
    from cinemateca.search.types import SearchResult

    _write_fake_clap_index(tmp_path, n=8, dim=4)
    idx = load_audio_index(tmp_path / "audio")
    assert idx is not None

    result = find_audio(idx, _RowStub(idx), "anything", film_slug="jeca_tatu", top_k=3)

    assert isinstance(result, SearchResult)
    assert result.retriever_mode == "audio"
    assert result.fusion_used is False
    assert result.reranker_applied is False
    assert result.num_films_searched == 1
    assert result.latency_ms is not None and result.latency_ms >= 0.0
    assert result.no_index is False
    # Hits are typed and carry the per-film slug join key; top-1 is scene 0.
    assert len(result.hits) == 3
    assert result.hits[0].scene_id == 0
    assert all(h.film_slug == "jeca_tatu" for h in result.hits)
    # Scores monotonic descending (leaf ordering preserved through the lift).
    scores = [h.score for h in result.hits]
    assert scores == sorted(scores, reverse=True)


def test_aggregate_audio_counts_films_with_index(tmp_path: Path) -> None:
    """aggregate_audio walks the registry → num_films_searched == N films that
    actually had a CLAP index searched (here 2 of 3 registered)."""
    from cinemateca.config import load_config
    from cinemateca.library import register_film
    from cinemateca.search.audio import aggregate_audio
    from cinemateca.search.types import SearchResult

    cfg = load_config(project_root=tmp_path)
    library_dir = tmp_path / "library"
    library_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.library_dir = library_dir

    for slug in ("film_a", "film_b", "film_c"):
        register_film(library_dir, slug=slug, title=slug, year=1959, raw_filename=f"{slug}.mp4")
    # Only film_a and film_c get a CLAP index; film_b has none → skipped.
    _write_fake_clap_index(library_dir / "film_a", n=4, dim=4)
    _write_fake_clap_index(library_dir / "film_c", n=4, dim=4)

    # Stub factory: returns an embedder whose query is a fixed unit vector.
    class _UnitStub:
        def encode_text(self, text: str) -> np.ndarray:
            v = np.ones(4, dtype="float32")
            return v / np.linalg.norm(v)

    calls = {"n": 0}

    def _factory(_cfg) -> _UnitStub:
        calls["n"] += 1
        return _UnitStub()

    result = aggregate_audio(cfg, _factory, "anything", top_k=10)

    assert isinstance(result, SearchResult)
    assert result.retriever_mode == "audio"
    assert result.fusion_used is False
    assert result.reranker_applied is False
    assert result.num_films_searched == 2
    assert result.no_index is False
    assert result.latency_ms is not None and result.latency_ms >= 0.0
    # Embedder built at most once across the walk.
    assert calls["n"] == 1
    slugs = {h.film_slug for h in result.hits}
    assert slugs == {"film_a", "film_c"}


def test_aggregate_audio_no_index_when_no_film_has_clap(tmp_path: Path) -> None:
    """No registered film carries a CLAP index → no_index True,
    num_films_searched 0, and the embedder factory is never invoked."""
    from cinemateca.config import load_config
    from cinemateca.library import register_film
    from cinemateca.search.audio import aggregate_audio

    cfg = load_config(project_root=tmp_path)
    library_dir = tmp_path / "library"
    library_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.library_dir = library_dir
    register_film(library_dir, slug="film_a", title="A", year=1959, raw_filename="film_a.mp4")

    def _factory(_cfg):  # pragma: no cover - must not be called
        raise AssertionError("embedder factory must not be built when no index exists")

    result = aggregate_audio(cfg, _factory, "anything", top_k=5)
    assert result.no_index is True
    assert result.num_films_searched == 0
    assert result.retriever_mode == "audio"
    assert result.hits == []
