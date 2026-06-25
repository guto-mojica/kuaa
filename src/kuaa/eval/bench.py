"""Retrieval-latency benchmark core (E6).

Pure timing + stats logic extracted from ``scripts/bench_retrieval.py``.
No ``api.*`` imports — this module is the reusable library function
consumed by the WS-5 T9 CI job and by the eval pipeline.

The script keeps:
  * ``BenchFixture`` dataclass + ``_build_fixture`` (api.services imports)
  * Arg parsing, hardware probe, JSON/Markdown writers
  * ``main()``

This module provides:
  * ``_percentile(samples, pct)`` — nearest-rank percentile
  * ``summarize(samples_ms)`` — canonical stats dict
  * ``BenchResult`` — typed result container
  * ``bench_retrievers(fx, queries, *, top_k)`` — timed loop + aggregation
  * ``_time_clip / _time_bm25 / _time_hybrid / _warmup`` — per-query helpers

The per-stage timing in ``_time_hybrid`` is re-keyed onto the ``timed``
context manager (F5 hook, ``kuaa.timing``) rather than raw
``time.perf_counter()`` deltas.  This makes the latency data available
to any consumer that hooks into ``timed`` (e.g. structured logging).
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from kuaa.timing import timed

# ─── Constants ────────────────────────────────────────────────────────────────

WARMUP_QUERIES = 5
RRF_K = 60  # matches DEFAULT_RRF_K in the production dispatcher
SEM_W = 0.70  # matches config/default.yaml → search.hybrid_sem_w
BM25_W = 0.30  # matches config/default.yaml → search.hybrid_bm25_w


# ─── Stats helpers ────────────────────────────────────────────────────────────


def _percentile(samples: list[float], pct: float) -> float:
    """Nearest-rank percentile (Wikipedia "ordinal" definition).

    p50 of [1, 2, 3] = 2. p100 = max(samples). p0 = min(samples).
    Empty input is undefined — caller guards against it.
    """
    if not samples:
        return math.nan
    s = sorted(samples)
    if pct <= 0:
        return s[0]
    if pct >= 100:
        return s[-1]
    # Nearest-rank: position = ceil(pct/100 * N); 1-indexed.
    pos = max(1, math.ceil(pct / 100.0 * len(s)))
    return s[pos - 1]


def summarize(samples_ms: list[float]) -> dict:
    """Return the canonical stats dict for a vector of millisecond samples.

    Keys: n, p50, p95, p99, mean, max.  All values are None when the
    input is empty, so callers can distinguish "no data" from zero.
    """
    if not samples_ms:
        return {"n": 0, "p50": None, "p95": None, "p99": None, "mean": None, "max": None}
    return {
        "n": len(samples_ms),
        "p50": _percentile(samples_ms, 50),
        "p95": _percentile(samples_ms, 95),
        "p99": _percentile(samples_ms, 99),
        "mean": statistics.fmean(samples_ms),
        "max": max(samples_ms),
    }


# ─── Result container ─────────────────────────────────────────────────────────


@dataclass
class BenchResult:
    """Aggregated result from ``bench_retrievers``.

    ``modes`` maps retriever name → ``summarize()`` dict.
    ``hybrid_stages`` maps sub-stage name → ``summarize()`` dict.
    ``raw_samples_ms`` keeps per-query millisecond lists for re-analysis.
    ``n_queries``, ``top_k``, ``raw_k``, ``loop_wall_s`` mirror the
    script's existing JSON payload keys so ``as_dict()`` preserves the
    output contract.
    """

    n_queries: int
    top_k: int
    raw_k: int
    loop_wall_s: float
    modes: dict[str, dict]
    hybrid_stages: dict[str, dict]
    raw_samples_ms: dict[str, list[float]]

    def as_dict(self) -> dict:
        """Return a dict that matches the ``results`` sub-key of the JSON payload.

        The script's ``write_json`` and ``write_markdown`` consume
        ``payload["results"]`` — this method produces an identical structure
        to the old ``run_bench`` return value so the writers need no changes.
        """
        return {
            "n_queries": self.n_queries,
            "top_k": self.top_k,
            "raw_k": self.raw_k,
            "loop_wall_s": self.loop_wall_s,
            "modes": self.modes,
            "hybrid_stages": self.hybrid_stages,
            "raw_samples_ms": self.raw_samples_ms,
        }


# ─── Per-query timed helpers ──────────────────────────────────────────────────


def _time_clip(fx: object, query: str, *, raw_k: int, top_k: int) -> float:
    """Total ms for a CLIP-only query through production ``search_text``."""
    from kuaa.search.clip import search_text

    # fx is BenchFixture (duck-typed to avoid importing it from the script)
    with timed("clip_search") as t:
        _ = search_text(fx.index, query, [], {}, top_k, fx.min_similarity)  # type: ignore[attr-defined]
    return t.elapsed_ms


def _time_bm25(fx: object, query: str, *, raw_k: int, top_k: int) -> float:
    """Total ms for a BM25-only query."""
    with timed("bm25_query") as t:
        _ = fx.bm25.query(query, top_k=raw_k)  # type: ignore[attr-defined]
    return t.elapsed_ms


def _time_hybrid(fx: object, query: str, *, raw_k: int, top_k: int) -> dict:
    """Total + 4 sub-stage timings for one hybrid query, re-keyed onto timed.

    Returns ``{"total": ms, "clip_best_row": ms, "clip_search": ms,
    "bm25_query": ms, "rrf_materialize": ms}``.

    Sub-stages map onto the same four sequential calls as the production
    ``search_hybrid`` dispatcher:
      1. clip_best_row    — ``_best_row_by_sid_from_embeddings``
      2. clip_search      — ``search_text``
      3. bm25_query       — ``BM25Index.query``
      4. rrf_materialize  — ``fuse_rrf`` + ``_fused_to_dataframe``
    """
    import time

    from kuaa.retrieval.hybrid import fuse_rrf
    from kuaa.search.clip import search_text
    from kuaa.search.hybrid import (
        _best_row_by_sid_from_embeddings,
        _fused_to_dataframe,
    )

    t0 = time.perf_counter()

    with timed("clip_best_row") as t_a:
        best_row_by_sid = _best_row_by_sid_from_embeddings(fx.index, query)  # type: ignore[attr-defined]

    with timed("clip_search") as t_b:
        clip_df = search_text(fx.index, query, [], {}, raw_k, fx.min_similarity)  # type: ignore[attr-defined]
        clip_ranked: list[tuple[int, float]] = (
            [(int(row.scene_id), float(row.similarity)) for row in clip_df.itertuples(index=False)]
            if not clip_df.empty
            else []
        )

    with timed("bm25_query") as t_c:
        bm25_hits = fx.bm25.query(query, top_k=raw_k)  # type: ignore[attr-defined]

    with timed("rrf_materialize") as t_d:
        fused = fuse_rrf(clip_ranked, bm25_hits, sem_w=SEM_W, bm25_w=BM25_W, k_rrf=RRF_K)[:top_k]
        _ = _fused_to_dataframe(
            fused,
            clip_df,
            fx.index,  # type: ignore[attr-defined]
            [],
            {},
            top_k,
            best_row_by_sid=best_row_by_sid,
        )

    total = (time.perf_counter() - t0) * 1000.0
    return {
        "total": total,
        "clip_best_row": t_a.elapsed_ms,
        "clip_search": t_b.elapsed_ms,
        "bm25_query": t_c.elapsed_ms,
        "rrf_materialize": t_d.elapsed_ms,
    }


def _warmup(fx: object, *, raw_k: int, top_k: int) -> None:
    """Run ``WARMUP_QUERIES`` throwaway queries to prime caches/JIT."""
    warm_queries = [
        "two men talking",
        "a horse running through a field",
        "interior office scene",
        "close-up of a face",
        "outdoor crowd shot",
    ]
    for q in warm_queries[:WARMUP_QUERIES]:
        _time_hybrid(fx, q, raw_k=raw_k, top_k=top_k)


# ─── Main bench loop ──────────────────────────────────────────────────────────


def bench_retrievers(fx: object, queries: list[str], *, top_k: int) -> BenchResult:
    """Run the full timed loop and return a ``BenchResult``.

    Duck-typed ``fx``: must have ``.index``, ``.bm25``, ``.min_similarity``
    attributes compatible with ``BenchFixture``.  The function intentionally
    does not import ``BenchFixture`` (which lives in the script and pulls
    ``api.services.*``) so this module stays ``api.*``-free.

    Args:
        fx:      Fixture object (duck-typed BenchFixture).
        queries: List of query strings to time.
        top_k:   Number of results per query (passed to retrievers).

    Returns:
        ``BenchResult`` with per-mode + per-stage stats + raw samples.
    """
    import time as _time

    # Mirrors search_hybrid's 4× widening before raw_k is passed to
    # search_text and BM25.
    raw_k = max(top_k * 4, 1)

    print(f"  warm-up: {WARMUP_QUERIES} throwaway queries…", flush=True)
    _warmup(fx, raw_k=raw_k, top_k=top_k)

    print(f"  timing {len(queries)} queries x 3 modes (k={top_k}, raw_k={raw_k})…", flush=True)
    clip_ms: list[float] = []
    bm25_ms: list[float] = []
    hybrid_total_ms: list[float] = []
    stage_ms: dict[str, list[float]] = {
        "clip_best_row": [],
        "clip_search": [],
        "bm25_query": [],
        "rrf_materialize": [],
    }

    t_loop = _time.perf_counter()
    for q in queries:
        clip_ms.append(_time_clip(fx, q, raw_k=raw_k, top_k=top_k))
        bm25_ms.append(_time_bm25(fx, q, raw_k=raw_k, top_k=top_k))
        h = _time_hybrid(fx, q, raw_k=raw_k, top_k=top_k)
        hybrid_total_ms.append(h["total"])
        for stage, val in h.items():
            if stage == "total":
                continue
            stage_ms[stage].append(val)
    loop_wall = _time.perf_counter() - t_loop

    return BenchResult(
        n_queries=len(queries),
        top_k=top_k,
        raw_k=raw_k,
        loop_wall_s=loop_wall,
        modes={
            "clip": summarize(clip_ms),
            "bm25": summarize(bm25_ms),
            "hybrid": summarize(hybrid_total_ms),
        },
        hybrid_stages={stage: summarize(vals) for stage, vals in stage_ms.items()},
        raw_samples_ms={
            "clip": clip_ms,
            "bm25": bm25_ms,
            "hybrid_total": hybrid_total_ms,
            "hybrid_clip_best_row": stage_ms["clip_best_row"],
            "hybrid_clip_search": stage_ms["clip_search"],
            "hybrid_bm25_query": stage_ms["bm25_query"],
            "hybrid_rrf_materialize": stage_ms["rrf_materialize"],
        },
    )


__all__ = [
    "BenchResult",
    "WARMUP_QUERIES",
    "RRF_K",
    "SEM_W",
    "BM25_W",
    "_percentile",
    "summarize",
    "bench_retrievers",
    "_time_clip",
    "_time_bm25",
    "_time_hybrid",
    "_warmup",
]
