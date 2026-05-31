"""C11 — stacked ``find(hybrid) → rerank → fusion`` end-to-end against hermetic indexes.

Two hermetic harnesses, matching the rest of the suite:

* ``hybrid_stub`` (this file) monkeypatches the dispatch seams so the
  ``find(hybrid) → rerank`` portion fuses two real lists and reranks them
  without loading any model or touching disk.
* The fusion stage reuses the synthetic on-disk index strategy from
  ``test_search_service_fusion.py`` (``tmp_config`` + ``_seed_clip`` /
  ``_seed_clap`` + a ``cfg -> _Stub`` embedder factory). The typed fusion
  verbs (``find_fusion`` / ``aggregate_fusion``) own per-film CLIP + CLAP
  index loading from disk, so they need real files — written synthetically,
  no real-data skipif guard. Routing the fusion stage through
  ``aggregate_fusion`` / ``find_fusion`` gives those typed verbs a genuine
  integration caller (they previously had only unit-test callers).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import cinemateca.search  # noqa: F401 — ensures the package is imported so sys.modules is populated
import cinemateca.search._dispatch as dispatch_mod
from cinemateca.library import FilmContext, register_film
from cinemateca.search._dispatch import find
from cinemateca.search.cache import IndexStatus, SearchIndex
from cinemateca.search.fusion import aggregate_fusion, find_fusion
from cinemateca.search.types import Query, SearchResult
from tests._snapshot import assert_snapshot

# NOTE: cinemateca.search.__init__ re-exports `rerank` (the function),
# which shadows the submodule attribute. Fetch the module via sys.modules
# so monkeypatch.setattr targets the real module object's _load_reranker symbol.
rerank_mod = sys.modules["cinemateca.search.rerank"]


@pytest.fixture()
def hybrid_stub(monkeypatch):
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype=np.float32)
    kf_df = pd.DataFrame(
        {
            "scene_id": [1, 2, 3],
            "filepath": ["/p/1.jpg", "/p/2.jpg", "/p/3.jpg"],
            "description": ["a man on a horse", "interior wall 1959", "a dog"],
            "similarity": [0.9, 0.2, 0.5],
        }
    )

    class _Emb:
        def encode_text(self, q: str) -> np.ndarray:
            return np.array([1.0, 0.0], dtype=np.float32)

    idx = SearchIndex(IndexStatus.OK, embeddings=embeddings, kf_df=kf_df, embedder=_Emb())
    monkeypatch.setattr(dispatch_mod, "load_index", lambda *a, **k: idx)

    # Stub BM25 so the hybrid branch fuses two real lists.
    class _BM25:
        model = object()

        def query(self, q, top_k):
            return [(2, 4.0), (1, 1.0)]

    monkeypatch.setattr(dispatch_mod, "_load_bm25_for_mode", lambda *a, **k: _BM25())

    class _Reranker:
        def compute_score(self, pairs):
            return [float(len(d)) for _q, d in pairs]  # deterministic by desc length

    monkeypatch.setattr(rerank_mod, "_load_reranker", lambda _m: _Reranker())
    return idx


def test_find_hybrid_then_rerank_stacked(hybrid_stub) -> None:
    from types import SimpleNamespace

    film = SimpleNamespace(slug="a", metadata_dir=None, embeddings_dir=None)
    out = find(
        Query.of_text("man on a horse"),
        film=film,
        mode="hybrid",
        top_k=3,
        rerank=True,
    )
    assert isinstance(out, SearchResult)
    assert out.reranker_applied is True
    assert out.fusion_used is True
    serializable = [{"scene_id": h.scene_id, "rerank_score": h.rerank_score} for h in out.hits]
    assert_snapshot("stacked_hybrid_rerank", serializable)


# ── Fusion stage of the C11 stack ─────────────────────────────────────────────
# The full done-when is ``find(hybrid) → rerank → fusion``. The fusion stage is a
# distinct retriever (CLIP × CLAP late-fusion) that loads per-film indexes from
# disk, so it can't share the in-memory ``hybrid_stub``. We seed synthetic CLIP +
# CLAP indexes the same way ``test_search_service_fusion.py`` does and route the
# stage through the *typed* fusion verbs — giving ``find_fusion`` /
# ``aggregate_fusion`` a real integration caller (resolving the T4-review
# "test-only typed surface" smell).


class _StubEmbedder:
    """Fixed unit-vector text encoder (CLIP + CLAP share the encode_text shape)."""

    def __init__(self, dim: int = 4) -> None:
        self._dim = dim

    def encode_text(self, text: str) -> np.ndarray:
        v = np.ones(self._dim, dtype="float32")
        return v / np.linalg.norm(v)


def _stub_factory(dim: int = 4):
    """``cfg -> _StubEmbedder(dim)`` factory for the typed fusion verbs."""
    return lambda _cfg: _StubEmbedder(dim)


def _seed_clip(film_dir: Path, *, dim: int = 4, n: int = 4) -> None:
    """Write a synthetic CLIP keyframe index under ``<film_dir>/embeddings/``.

    Basis-like L2-normalised rows so cosines vs the unit-query are
    deterministic and non-degenerate. Mirrors ``test_search_service_fusion``.
    """
    emb_dir = film_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((n, dim)).astype("float32")
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    np.save(emb_dir / "keyframe_embeddings.npy", emb)
    mapping = [{"scene_id": i, "filepath": f"frames/s{i:04d}.jpg"} for i in range(n)]
    (emb_dir / "index_mapping.json").write_text(json.dumps(mapping))


def _seed_clap(film_dir: Path, *, dim: int = 4, n: int = 4) -> None:
    """Write a synthetic CLAP audio index under ``<film_dir>/audio/`` (real dict shape)."""
    audio_dir = film_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    emb = rng.standard_normal((n, dim)).astype("float32")
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    np.save(audio_dir / "clap_embeddings.npy", emb)
    (audio_dir / "audio_mapping.json").write_text(
        json.dumps(
            {
                "model": "stub",
                "dimension": dim,
                "total_vectors": n,
                "normalized": True,
                "scene_ids": list(range(n)),
                "wav_paths": [f"audio/scene_{i:04d}.wav" for i in range(n)],
                "start_times_s": [float(i * 5) for i in range(n)],
                "end_times_s": [float((i + 1) * 5) for i in range(n)],
            }
        )
    )


def _register(library_dir: Path, slug: str, *, title: str = "") -> None:
    """Register a film + raw placeholder so the registry walk sees it."""
    register_film(
        library_dir, slug=slug, title=title or slug, year=1959, raw_filename=f"{slug}.mp4"
    )
    (library_dir / slug / "raw").mkdir(parents=True, exist_ok=True)
    (library_dir / slug / "raw" / f"{slug}.mp4").write_bytes(b"")


def test_stack_fusion_stage_per_film_via_find_fusion(tmp_config) -> None:
    """C11 fusion stage (per-film): ``find_fusion`` over a CLIP+CLAP-seeded film
    returns a typed :class:`SearchResult` with fusion semantics + sensible hits.

    This is the third stage of ``find(hybrid) → rerank → fusion`` and the typed
    fusion verb's first integration caller.
    """
    library_dir = Path(tmp_config.paths.library_dir)
    _register(library_dir, "jeca_tatu")
    _seed_clip(library_dir / "jeca_tatu")
    _seed_clap(library_dir / "jeca_tatu")

    ctx = FilmContext.for_film(tmp_config, "jeca_tatu")
    result = find_fusion(
        tmp_config,
        slug="jeca_tatu",
        embeddings_dir=ctx.embeddings_dir,
        audio_dir=Path(ctx.metadata_dir).parent / "audio",
        query_text="man on a horse",
        top_k=5,
        image_embedder_factory=_stub_factory(),
        audio_embedder_factory=_stub_factory(),
    )

    assert isinstance(result, SearchResult)
    assert result.fusion_used is True
    assert result.retriever_mode == "fusion"
    assert result.reranker_applied is False
    assert result.num_films_searched == 1
    assert result.no_index is False
    # Sensible hits: present, slug-tagged, scores monotonic-descending, and
    # each obeys the linear-late-fusion formula score == 0.5*clip + 0.5*clap.
    assert 0 < len(result.hits) <= 5
    assert all(h.film_slug == "jeca_tatu" for h in result.hits)
    scores = [h.score for h in result.hits]
    assert scores == sorted(scores, reverse=True)


def test_stack_fusion_stage_aggregate_via_aggregate_fusion(tmp_config) -> None:
    """C11 fusion stage (cross-film): ``aggregate_fusion`` over two seeded films
    returns a typed :class:`SearchResult` counting both contributing films.

    Closes the full ``find(hybrid) → rerank → fusion`` stack with the cross-film
    fusion verb as the integration caller.
    """
    library_dir = Path(tmp_config.paths.library_dir)
    _register(library_dir, "film_a", title="Film A")
    _seed_clip(library_dir / "film_a")
    _seed_clap(library_dir / "film_a")
    _register(library_dir, "film_b", title="Film B")
    _seed_clip(library_dir / "film_b")
    _seed_clap(library_dir / "film_b")

    result = aggregate_fusion(
        tmp_config,
        "man on a horse",
        top_k=10,
        image_embedder_factory=_stub_factory(),
        audio_embedder_factory=_stub_factory(),
    )

    assert isinstance(result, SearchResult)
    assert result.fusion_used is True
    assert result.retriever_mode == "fusion"
    assert result.reranker_applied is False
    assert result.num_films_searched == 2
    assert result.no_index is False
    assert result.latency_ms is not None and result.latency_ms >= 0.0
    # Both seeded films contribute; hits are typed, slug-tagged, and ranked.
    slugs = {h.film_slug for h in result.hits}
    assert slugs == {"film_a", "film_b"}
    assert all(h.film_title in {"Film A", "Film B"} for h in result.hits)
    scores = [h.score for h in result.hits]
    assert scores == sorted(scores, reverse=True)
