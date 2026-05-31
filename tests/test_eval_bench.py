"""Tests for cinemateca.eval.bench — TDD for E6.

Two test cases:
  1. test_summarize_percentiles  — nearest-rank percentile contract.
  2. test_bench_retrievers_uses_timed_hook — hermetic duck-typed fixture,
     checks BenchResult shape and timed labels.
"""

from __future__ import annotations

import math
from unittest.mock import patch

import pytest

# ─── test 1: summarize() ──────────────────────────────────────────────────────


def test_summarize_percentiles() -> None:
    """Nearest-rank percentile contract for summarize([1,2,3,4]).

    Expected values derived from the _percentile() nearest-rank definition:
      p50: ceil(0.50*4)=2 → s[1] = 2.0
      p95: ceil(0.95*4)=ceil(3.8)=4 → s[3] = 4.0
      p99: ceil(0.99*4)=ceil(3.96)=4 → s[3] = 4.0
      mean: 2.5
      max: 4.0
      n: 4
    """
    from cinemateca.eval.bench import summarize

    s = summarize([1.0, 2.0, 3.0, 4.0])
    assert s["n"] == 4
    assert s["p50"] == 2.0
    assert s["p95"] == 4.0
    assert s["p99"] == 4.0
    assert abs(s["mean"] - 2.5) < 1e-9
    assert s["max"] == 4.0


def test_summarize_empty() -> None:
    from cinemateca.eval.bench import summarize

    s = summarize([])
    assert s["n"] == 0
    assert s["p50"] is None


# ─── test 2: bench_retrievers() uses timed hook ───────────────────────────────


def test_bench_retrievers_uses_timed_hook(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hermetic test: fake fixture + monkeypatched timed.

    Verifies:
      * BenchResult.modes["hybrid"]["p50"] is a float (not None)
      * BenchResult.modes["clip"]["p50"] is a float
      * BenchResult.modes["bm25"]["p50"] is a float
      * timed was called with labels including clip_search, bm25_query,
        rrf_materialize (the three canonical sub-stage labels that the
        benchmark harness documents).
    """
    import cinemateca.eval.bench as bench_mod

    # ── recorded timed labels ─────────────────────────────────────────────
    recorded_labels: list[str] = []

    # Build a context-manager replacement that records labels and returns
    # a Timer-like object with elapsed_ms=1.0 so tests are deterministic.
    from contextlib import contextmanager
    from dataclasses import dataclass as _dc

    @_dc
    class _FakeTimer:
        label: str | None = None
        elapsed_ms: float = 1.0

    @contextmanager
    def _fake_timed(label: str | None = None):
        if label:
            recorded_labels.append(label)
        yield _FakeTimer(label=label, elapsed_ms=1.0)

    monkeypatch.setattr(bench_mod, "timed", _fake_timed)

    # ── fake CLIP search results (empty DataFrame) ────────────────────────
    import pandas as pd

    _empty_df = pd.DataFrame(columns=["scene_id", "film_slug", "similarity", "keyframe_url"])

    # ── minimal duck-typed fixture attrs needed by _time_* functions ──────
    # Read bench_mod source to learn what each timing helper accesses:
    #   _time_clip: fx.index, fx.min_similarity → search_text(fx.index, q, [], {}, raw_k, fx.min_similarity)
    #   _time_bm25: fx.bm25 → bm25.query(q, top_k=raw_k)
    #   _time_hybrid: fx.index, fx.bm25, fx.min_similarity → _best_row_by_sid, search_text, bm25.query, fuse_rrf, _fused_to_dataframe

    class _FakeBM25:
        scene_ids: list[int] = []

        def query(self, text: str, *, top_k: int) -> list[tuple[int, float]]:
            return [(1, 0.5), (2, 0.3)]

    class _FakeEmbedder:
        pass

    class _FakeIndex:
        embedder = _FakeEmbedder()
        embeddings = None
        kf_df = _empty_df
        status = None  # not checked in timing helpers

    class _FakeFx:
        slug = "test_film"
        n_scenes = 3
        n_vectors = 5
        n_bm25_docs = 3
        device = "cpu"
        min_similarity = 0.0
        embedder = _FakeEmbedder()
        index = _FakeIndex()
        bm25 = _FakeBM25()

    fx = _FakeFx()
    queries = ["rain on a rooftop", "two actors talking", "sunset over mountains"]

    # Patch the search functions that the timing helpers call so they return
    # empty/trivial values without hitting any model.
    with (
        patch(
            "cinemateca.search.clip.search_text",
            return_value=_empty_df,
        ),
        patch(
            "cinemateca.search.hybrid._best_row_by_sid_from_embeddings",
            return_value={},
        ),
        patch(
            "cinemateca.search.hybrid._fused_to_dataframe",
            return_value=_empty_df,
        ),
        patch(
            "cinemateca.retrieval.hybrid.fuse_rrf",
            return_value=[],
        ),
    ):
        result = bench_mod.bench_retrievers(fx, queries, top_k=10)

    # ── structural assertions ─────────────────────────────────────────────
    assert hasattr(result, "modes"), "BenchResult must have .modes"
    assert "hybrid" in result.modes
    assert "clip" in result.modes
    assert "bm25" in result.modes

    hybrid_p50 = result.modes["hybrid"]["p50"]
    assert isinstance(hybrid_p50, float) and not math.isnan(
        hybrid_p50
    ), f"modes['hybrid']['p50'] should be a float, got {hybrid_p50!r}"

    clip_p50 = result.modes["clip"]["p50"]
    assert isinstance(clip_p50, float) and not math.isnan(
        clip_p50
    ), f"modes['clip']['p50'] should be a float, got {clip_p50!r}"

    bm25_p50 = result.modes["bm25"]["p50"]
    assert isinstance(bm25_p50, float) and not math.isnan(
        bm25_p50
    ), f"modes['bm25']['p50'] should be a float, got {bm25_p50!r}"

    # ── timed label assertions ─────────────────────────────────────────────
    assert (
        "clip_search" in recorded_labels
    ), f"expected 'clip_search' in timed labels; got: {recorded_labels}"
    assert (
        "bm25_query" in recorded_labels
    ), f"expected 'bm25_query' in timed labels; got: {recorded_labels}"
    assert (
        "rrf_materialize" in recorded_labels
    ), f"expected 'rrf_materialize' in timed labels; got: {recorded_labels}"
