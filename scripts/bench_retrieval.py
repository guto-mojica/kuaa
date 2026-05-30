#!/usr/bin/env python3
"""Retrieval-latency benchmark for the hybrid search pipeline.

Times the three retriever modes (clip / bm25 / hybrid) against one
already-indexed film and reports p50 / p95 / p99 / mean / max per mode.
For ``hybrid`` it additionally breaks the per-query latency into the same
sequential calls made by the production dispatcher:

  1. ``clip_best_row``     — ``_best_row_by_sid_from_embeddings(index, query)``
  2. ``clip_search``       — ``search_text(index, query, ..., raw_k, min_sim)``
  3. ``bm25_query``        — ``BM25Index.query(query, top_k=raw_k)``
  4. ``rrf_materialize``   — ``fuse_rrf(...)`` + ``_fused_to_dataframe(...)``

The first two CLIP stages each encode the query, matching the current
``search_hybrid`` implementation exactly. The breakdown is a strictly
additive instrumentation, no behaviour change.

Usage::

    uv run python scripts/bench_retrieval.py
    uv run python scripts/bench_retrieval.py --n 200 --k 50 --film jeca_tatu
    uv run python scripts/bench_retrieval.py --out data/perf/run_2026-05-24.json

    # Smoke mode (CI / fast local check — 8 queries, k=20):
    uv run python scripts/bench_retrieval.py --smoke
    uv run python scripts/bench_retrieval.py --smoke --film jeca_tatu

Outputs both a JSON results file (defaults to ``data/perf/bench_results.json``)
and a Markdown summary (``docs/PERFORMANCE.md``). The Markdown file
captures hardware, headline number, and ready-to-paste README snippet —
the JSON file is the raw record (per-mode + per-stage stats + every
sample, for re-analysis).

When no on-disk index is available (e.g. in CI without the demo bundle),
the script prints a notice and exits 0 rather than crashing.  The
``--smoke`` flag sets small defaults (``--n 8 --k 20``) so CI-level runs
are fast when an index IS present.

NB: warm-up of 5 throwaway queries primes the CLIP forward pass and the
BM25 cache before the timed loop starts, so the first call's compile/
JIT / lazy-load cost never leaks into p50/p95.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))


# ─── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_QUERIES_YAML = REPO_ROOT / "data" / "eval" / "archive_demo_queries.yaml"
DEFAULT_OUT_JSON = REPO_ROOT / "data" / "perf" / "bench_results.json"
DEFAULT_OUT_MD = REPO_ROOT / "docs" / "PERFORMANCE.md"
WARMUP_QUERIES = 5
RRF_K = 60  # matches DEFAULT_RRF_K
SEM_W = 0.70  # matches config/default.yaml → search.hybrid_sem_w
BM25_W = 0.30  # matches config/default.yaml → search.hybrid_bm25_w


# ─── Hardware probe ───────────────────────────────────────────────────────────


def _cpu_model() -> str:
    """Return a short CPU model string from /proc/cpuinfo, or platform fall-back."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except (FileNotFoundError, OSError):
        pass
    return platform.processor() or platform.machine() or "unknown CPU"


def _cpu_count() -> int:
    return os.cpu_count() or 1


def _gpu_info() -> str | None:
    """Return ``"<name>, <mem>"`` from nvidia-smi or ``None`` if no GPU."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            timeout=4,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    text = out.decode("utf-8", errors="replace").strip()
    return text or None


def _torch_device_info() -> dict:
    import torch

    info = {
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_name": None,
    }
    if info["cuda_available"]:
        info["device_name"] = torch.cuda.get_device_name(0)
    return info


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


def _summary(samples_ms: list[float]) -> dict:
    """Return the canonical stats dict for a vector of millisecond samples."""
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


# ─── Query pool ───────────────────────────────────────────────────────────────


def _seed_queries_from_yaml(path: Path) -> list[str]:
    """Read the ``text:`` field from every entry in the eval YAML."""
    if not path.exists():
        return []
    import yaml

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    out: list[str] = []
    for q in data.get("queries", []) or []:
        t = q.get("text")
        if isinstance(t, str) and t.strip():
            out.append(t.strip())
    return out


def _synthesise_from_descriptions(metadata_dir: Path, *, target: int) -> list[str]:
    """Pull short query strings from ``scene_descriptions.json`` so the pool
    looks like real user input on the same corpus.

    Heuristic: split each scene's ``description`` field on full-stops and
    keep the first sentence (capped at 12 words). This produces queries
    that resemble what an archivist would type ("Two women in traditional
    clothing stand…"), not random tokens. Returns at most ``target``
    distinct strings.
    """
    p = metadata_dir / "scene_descriptions.json"
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for entry in data:
        desc = (entry.get("description") or "").strip()
        if not desc:
            continue
        first = desc.split(".", 1)[0].strip()
        words = first.split()
        if len(words) < 3:
            continue
        clipped = " ".join(words[:12]).rstrip(",;:").strip()
        if not clipped or clipped.lower() in seen:
            continue
        seen.add(clipped.lower())
        out.append(clipped)
        if len(out) >= target:
            break
    return out


def build_query_pool(*, n: int, queries_yaml: Path, metadata_dir: Path) -> tuple[list[str], dict]:
    """Build ``n`` query strings + provenance dict for the report header.

    Strategy:
      1. Seed with every query in the eval YAML (~20–25 strings).
      2. Top up from scene descriptions (gives queries with real signal).
      3. Cycle the seed pool if we still need more.

    Returns ``(queries, provenance)`` where ``provenance`` is a small dict
    describing how the pool was assembled (logged in the JSON output for
    auditability of the numbers).
    """
    seed = _seed_queries_from_yaml(queries_yaml)
    syn: list[str] = []
    if len(seed) < n:
        syn = _synthesise_from_descriptions(metadata_dir, target=n - len(seed))
    pool = list(seed) + list(syn)
    # Top up by cycling if still short.
    cycled = 0
    while len(pool) < n and seed:
        pool.append(seed[cycled % len(seed)])
        cycled += 1
    queries = pool[:n]
    provenance = {
        "n_requested": n,
        "from_yaml": len(seed),
        "from_descriptions": len(syn),
        "cycled_repeats": cycled,
        "queries_yaml": (
            str(queries_yaml.relative_to(REPO_ROOT))
            if queries_yaml.is_relative_to(REPO_ROOT)
            else str(queries_yaml)
        ),
    }
    return queries, provenance


# ─── Bench fixtures ───────────────────────────────────────────────────────────


@dataclass
class BenchFixture:
    """Everything the timed loop needs, loaded once before warm-up."""

    slug: str
    n_scenes: int
    n_vectors: int
    n_bm25_docs: int
    embedder: object  # cinemateca.models.base.ImageEmbedder
    index: object  # api.services.search.SearchIndex
    bm25: object  # cinemateca.retrieval.bm25.BM25Index
    device: str
    min_similarity: float
    library_dir: Path = field(repr=False)
    metadata_dir: Path = field(repr=False)


def _pick_film(cfg, *, requested: str | None) -> str:
    """Return the slug of the film we will benchmark against.

    Strategy:
      * ``--film`` overrides whenever the slug is registered and has an
        on-disk index;
      * otherwise pick the registered film with the LARGEST scene count
        (largest corpus = most interesting BM25 stats).
    """
    from cinemateca.library import scan_library

    library_dir = Path(cfg.paths.library_dir)
    films = list(scan_library(library_dir))
    if not films:
        raise SystemExit(f"No films registered in {library_dir}/films.json — nothing to benchmark.")

    by_slug = {f.slug: f for f in films}
    if requested:
        if requested not in by_slug:
            raise SystemExit(f"--film={requested!r} not registered. Choices: {sorted(by_slug)}")
        return requested

    # Largest film first (ties broken by slug for determinism).
    films.sort(key=lambda f: (-f.scene_count, f.slug))
    return films[0].slug


def _build_fixture(cfg, *, slug: str) -> BenchFixture:
    """Load every artefact the timed loop needs.

    Mirrors the production code paths:
      * SearchIndex through ``api.services.search._get_search_index``
      * BM25Index through ``_get_bm25_index_for_ctx`` (uses the lru_cache)
      * embedder is built with ``device_from_config(cfg)`` so we time on
        the same device the production server would.
    """
    from api.services.film_context import FilmContext

    from api.services.search import IndexStatus, _get_bm25_index_for_ctx, _get_search_index
    from cinemateca.device import device_from_config
    from cinemateca.models.registry import get_image_embedder
    from cinemateca.search.cache import SearchIndex

    device = device_from_config(cfg)
    device_str = str(device)

    idx = _get_search_index(cfg, slug)
    if idx.status is not IndexStatus.OK:
        raise SystemExit(
            f"Refusing to bench: CLIP index for {slug!r} is {idx.status.value} "
            f"({idx.detail}). Process the film first."
        )

    ctx = FilmContext.for_film(cfg, slug)
    bm25 = _get_bm25_index_for_ctx(ctx)
    if bm25 is None or bm25.model is None:
        raise SystemExit(
            f"Refusing to bench: BM25 corpus for {slug!r} is empty. "
            f"Are scene_descriptions.json + scene_tags.json present?"
        )

    embedder = get_image_embedder(cfg, device)
    # Force the lazy-load so warm-up is the only timing-sensitive cost left.
    load_model = getattr(embedder, "_load_model", None)
    if callable(load_model):
        load_model()

    # The cached index may contain a default-constructed embedder. Replace it
    # locally so the production search functions below use the configured
    # backend/device while preserving the loaded embeddings and keyframe map.
    idx = SearchIndex(
        status=idx.status,
        embeddings=idx.embeddings,
        kf_df=idx.kf_df,
        embedder=embedder,
        detail=idx.detail,
    )

    n_vectors = int(getattr(idx.embeddings, "shape", [0])[0])
    # idx.kf_df is typed as object on the dataclass to keep the AI core
    # uncoupled from pandas at type-check time; at runtime it's always
    # a DataFrame here (status is OK).
    n_scenes = (
        int(idx.kf_df["scene_id"].nunique())  # type: ignore[index, union-attr]
        if hasattr(idx.kf_df, "columns")
        else 0
    )
    n_bm25_docs = len(bm25.scene_ids)

    return BenchFixture(
        slug=slug,
        n_scenes=n_scenes,
        n_vectors=n_vectors,
        n_bm25_docs=n_bm25_docs,
        embedder=embedder,
        index=idx,
        bm25=bm25,
        device=device_str,
        min_similarity=float(getattr(cfg.embeddings, "min_similarity", 0.0) or 0.0),
        library_dir=Path(cfg.paths.library_dir),
        metadata_dir=ctx.metadata_dir,
    )


def _bm25_query(bm25, query: str, top_k: int) -> list[tuple[int, float]]:
    """One BM25 query → ranked list (delegates straight to the index)."""
    return bm25.query(query, top_k=top_k)


# ─── Per-query timed measurements ─────────────────────────────────────────────


def _time_clip(fx: BenchFixture, query: str, *, raw_k: int, top_k: int) -> float:
    """Total ms for a CLIP-only query through production ``search_text``."""
    from cinemateca.search.clip import search_text

    t0 = time.perf_counter()
    _ = search_text(fx.index, query, [], {}, top_k, fx.min_similarity)
    return (time.perf_counter() - t0) * 1000.0


def _time_bm25(fx: BenchFixture, query: str, *, raw_k: int, top_k: int) -> float:
    """Total ms for a BM25-only query."""
    t0 = time.perf_counter()
    _ = _bm25_query(fx.bm25, query, raw_k)
    return (time.perf_counter() - t0) * 1000.0


def _time_hybrid(fx: BenchFixture, query: str, *, raw_k: int, top_k: int) -> dict:
    """Total + 4 sub-stage timings for one hybrid query.

    Returns ``{"total": ms, "clip_best_row": ms, "clip_search": ms,
    "bm25_query": ms, "rrf_materialize": ms}``. The four sub-stage sums
    should be ≤ total (small difference = ``time.perf_counter()`` jitter
    + glue cost).
    """
    from cinemateca.retrieval.hybrid import fuse_rrf
    from cinemateca.search.clip import search_text
    from cinemateca.search.hybrid import (
        _best_row_by_sid_from_embeddings,
        _fused_to_dataframe,
    )

    t0 = time.perf_counter()

    t_a = time.perf_counter()
    best_row_by_sid = _best_row_by_sid_from_embeddings(fx.index, query)
    t_b = time.perf_counter()
    clip_df = search_text(fx.index, query, [], {}, raw_k, fx.min_similarity)
    clip_ranked: list[tuple[int, float]] = (
        [(int(row.scene_id), float(row.similarity)) for row in clip_df.itertuples(index=False)]
        if not clip_df.empty
        else []
    )
    t_c = time.perf_counter()
    bm25_hits = _bm25_query(fx.bm25, query, raw_k)
    t_d = time.perf_counter()
    fused = fuse_rrf(clip_ranked, bm25_hits, sem_w=SEM_W, bm25_w=BM25_W, k_rrf=RRF_K)[:top_k]
    _ = _fused_to_dataframe(
        fused,
        clip_df,
        fx.index,
        [],
        {},
        top_k,
        best_row_by_sid=best_row_by_sid,
    )
    t_e = time.perf_counter()

    total = (t_e - t0) * 1000.0
    return {
        "total": total,
        "clip_best_row": (t_b - t_a) * 1000.0,
        "clip_search": (t_c - t_b) * 1000.0,
        "bm25_query": (t_d - t_c) * 1000.0,
        "rrf_materialize": (t_e - t_d) * 1000.0,
    }


# ─── Warm-up ──────────────────────────────────────────────────────────────────


def _warmup(fx: BenchFixture, *, raw_k: int, top_k: int) -> None:
    """Run ``WARMUP_QUERIES`` throwaway queries through every code path.

    This primes:
      * the CLIP forward pass (CUDA kernels JIT-compile + cuDNN
        autotunes on first call);
      * the BM25 ``get_scores`` numpy buffers;
      * the per-film ``_cached_bm25_index`` lru_cache (already warm
        from the fixture build, but the call cost itself is also
        warmed here);
      * any module-level lazy imports inside ``fuse_rrf``.
    """
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


def run_bench(fx: BenchFixture, queries: list[str], *, top_k: int) -> dict:
    """Run the full timed loop and return a results dict ready for JSON dump."""
    # Mirrors search_hybrid's 4× widening before it passes raw_k into
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

    t_loop = time.perf_counter()
    for i, q in enumerate(queries):
        clip_ms.append(_time_clip(fx, q, raw_k=raw_k, top_k=top_k))
        bm25_ms.append(_time_bm25(fx, q, raw_k=raw_k, top_k=top_k))
        h = _time_hybrid(fx, q, raw_k=raw_k, top_k=top_k)
        hybrid_total_ms.append(h["total"])
        for stage, val in h.items():
            if stage == "total":
                continue
            stage_ms[stage].append(val)
    loop_wall = time.perf_counter() - t_loop

    return {
        "n_queries": len(queries),
        "top_k": top_k,
        "raw_k": raw_k,
        "loop_wall_s": loop_wall,
        "modes": {
            "clip": _summary(clip_ms),
            "bm25": _summary(bm25_ms),
            "hybrid": _summary(hybrid_total_ms),
        },
        "hybrid_stages": {stage: _summary(vals) for stage, vals in stage_ms.items()},
        # Keep raw samples around — small (3 lists × N floats) and lets
        # downstream analysis recompute different percentiles.
        "raw_samples_ms": {
            "clip": clip_ms,
            "bm25": bm25_ms,
            "hybrid_total": hybrid_total_ms,
            "hybrid_clip_best_row": stage_ms["clip_best_row"],
            "hybrid_clip_search": stage_ms["clip_search"],
            "hybrid_bm25_query": stage_ms["bm25_query"],
            "hybrid_rrf_materialize": stage_ms["rrf_materialize"],
        },
    }


# ─── Output: JSON + Markdown ──────────────────────────────────────────────────


def _fmt_ms(v: float | None) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    if v < 10:
        return f"{v:.2f}"
    if v < 100:
        return f"{v:.1f}"
    return f"{v:.0f}"


def write_json(payload: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"  wrote {out_path.relative_to(REPO_ROOT)}", flush=True)


def write_markdown(payload: dict, out_path: Path) -> None:
    """Write the PERFORMANCE.md report.

    Structure:
      1. Hardware + corpus header.
      2. Headline number (hybrid p95).
      3. Per-mode table.
      4. Per-stage breakdown (hybrid).
      5. README snippet block.
      6. Footer (how to re-run).
    """
    hw = payload["hardware"]
    fx = payload["fixture"]
    modes = payload["results"]["modes"]
    stages = payload["results"]["hybrid_stages"]
    top_k = payload["results"]["top_k"]
    n_q = payload["results"]["n_queries"]
    raw_k = payload["results"]["raw_k"]

    hybrid_p95 = modes["hybrid"]["p95"]

    headline = (
        f"**Hybrid retriever p95 = {_fmt_ms(hybrid_p95)} ms** "
        f"at k={top_k} on a {fx['n_scenes']}-scene corpus "
        f"({fx['n_vectors']} CLIP keyframe vectors, "
        f"{fx['n_bm25_docs']} BM25 documents)."
    )

    lines: list[str] = []
    lines.append("# Retrieval performance")
    lines.append("")
    lines.append(f"_Last benchmarked: {payload['timestamp']}_.")
    lines.append("")
    lines.append("## Hardware")
    lines.append("")
    lines.append(f"- CPU: {hw['cpu_model']} ({hw['cpu_count']} threads)")
    lines.append(f"- GPU: {hw['gpu'] or '_none — CPU-only_'}")
    lines.append(f"- PyTorch: {hw['torch_version']} (cuda available: {hw['cuda_available']})")
    lines.append(f"- Device used for CLIP encode: `{fx['device']}`")
    lines.append(f"- OS: {hw['platform']}")
    lines.append("")
    lines.append("## Corpus")
    lines.append("")
    lines.append(f"- Film: `{fx['slug']}` ({fx['n_scenes']} scenes)")
    lines.append(f"- CLIP keyframe vectors: {fx['n_vectors']}")
    lines.append(f"- BM25 documents (scenes with description or tags): {fx['n_bm25_docs']}")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(headline)
    lines.append("")
    lines.append(
        f"Measured across {n_q} queries (warm-up: {WARMUP_QUERIES} discarded), "
        f"each retriever fetches `raw_k={raw_k}` candidates per stage and the "
        f"dispatcher trims to `k={top_k}` (`min_similarity={fx['min_similarity']}`)."
    )
    lines.append("")

    # Per-mode table.
    lines.append("## Per-mode latency (ms)")
    lines.append("")
    lines.append("| Mode   | p50  | p95  | p99  | mean | max  |")
    lines.append("|--------|-----:|-----:|-----:|-----:|-----:|")
    for mode in ("clip", "bm25", "hybrid"):
        s = modes[mode]
        lines.append(
            f"| {mode:<6} | {_fmt_ms(s['p50']):>4} | {_fmt_ms(s['p95']):>4} | "
            f"{_fmt_ms(s['p99']):>4} | {_fmt_ms(s['mean']):>4} | {_fmt_ms(s['max']):>4} |"
        )
    lines.append("")

    # Per-stage breakdown.
    lines.append("## Hybrid sub-stage breakdown (ms)")
    lines.append("")
    lines.append("| Stage           | p50  | p95  | p99  | mean | max  |")
    lines.append("|-----------------|-----:|-----:|-----:|-----:|-----:|")
    for stage in ("clip_best_row", "clip_search", "bm25_query", "rrf_materialize"):
        s = stages[stage]
        lines.append(
            f"| {stage:<15} | {_fmt_ms(s['p50']):>4} | {_fmt_ms(s['p95']):>4} | "
            f"{_fmt_ms(s['p99']):>4} | {_fmt_ms(s['mean']):>4} | {_fmt_ms(s['max']):>4} |"
        )
    lines.append("")
    lines.append(
        "The four sub-stages mirror `cinemateca.search.hybrid.search_hybrid`: "
        "best-keyframe backfill, CLIP `search_text`, BM25 query, then RRF "
        "fusion plus DataFrame materialization. The first two stages each "
        "perform a text encode, matching the current dispatcher."
    )
    lines.append("")

    # README snippet (kept short so a reviewer can paste it as-is).
    lines.append("## README snippet")
    lines.append("")
    lines.append("```")
    lines.append(
        f"Hybrid retriever (CLIP + BM25 + weighted RRF): "
        f"p50 {_fmt_ms(modes['hybrid']['p50'])} ms, "
        f"p95 {_fmt_ms(modes['hybrid']['p95'])} ms, "
        f"p99 {_fmt_ms(modes['hybrid']['p99'])} ms"
    )
    lines.append(f"  on a {fx['n_scenes']}-scene corpus, k={top_k}, " f"device={fx['device']}")
    lines.append(
        f"  (CPU: {hw['cpu_model'].split('@')[0].strip()}"
        + (f"; GPU: {hw['gpu'].split(',')[0].strip()})" if hw["gpu"] else ")")
    )
    lines.append("```")
    lines.append("")

    # Footer — how to re-run.
    lines.append("## Re-running this benchmark")
    lines.append("")
    lines.append("```bash")
    lines.append(
        f"uv run python scripts/bench_retrieval.py --n {n_q} --k {top_k} --film {fx['slug']}"
    )
    lines.append("```")
    lines.append("")
    lines.append(
        "Per-query samples and the raw JSON live in "
        "`data/perf/bench_results.json` (rebuilt on every run). "
        "Numbers above reflect a warmed CLIP model + warmed BM25 cache; "
        "the first 5 queries of every run are discarded so they don't "
        "leak cold-start latency into the percentiles."
    )
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote {out_path.relative_to(REPO_ROOT)}", flush=True)


# ─── Entry point ──────────────────────────────────────────────────────────────


SMOKE_N = 8   # --smoke default: small query count for CI / quick local runs
SMOKE_K = 20  # --smoke default top_k


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--smoke",
        action="store_true",
        default=False,
        help=(
            f"Smoke mode: override --n to {SMOKE_N} and --k to {SMOKE_K} for a fast CI-friendly "
            "run.  Also guards gracefully when no on-disk index is present (prints a notice + "
            "exits 0).  Intended for CI jobs and quick local sanity checks; not a substitute for "
            "a full bench run."
        ),
    )
    p.add_argument("--n", type=int, default=100, help="Number of timed queries (default: 100).")
    p.add_argument("--k", type=int, default=50, help="top_k passed to retrievers (default: 50).")
    p.add_argument(
        "--film",
        default=None,
        help="Film slug to bench (default: largest registered film).",
    )
    p.add_argument(
        "--out",
        default=str(DEFAULT_OUT_JSON),
        help=f"JSON output path (default: {DEFAULT_OUT_JSON.relative_to(REPO_ROOT)}).",
    )
    p.add_argument(
        "--markdown",
        default=str(DEFAULT_OUT_MD),
        help=f"Markdown output path (default: {DEFAULT_OUT_MD.relative_to(REPO_ROOT)}).",
    )
    return p.parse_args(argv)


def _check_index_exists(cfg, *, film: str | None, smoke: bool) -> bool:
    """Return True if at least one registered film has an on-disk CLIP index.

    When ``film`` is specified, only that slug is checked.  Returns False
    (prints a ``::notice::`` + a human message) rather than raising, so callers
    can exit 0 gracefully — this is the CI no-op path.
    """
    try:
        from cinemateca.library import scan_library

        library_dir = Path(cfg.paths.library_dir)
        films = list(scan_library(library_dir))
    except Exception:
        return False
    if not films:
        return False
    if film is not None:
        for f in films:
            if f.slug == film:
                idx_path = Path(cfg.paths.library_dir) / film / "embeddings" / "clip_embeddings.npy"
                return idx_path.exists()
        return False
    # Any film with a CLIP index will do.
    for f in films:
        idx_path = (
            Path(cfg.paths.library_dir)
            / f.slug
            / "embeddings"
            / "clip_embeddings.npy"
        )
        if idx_path.exists():
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # --smoke: override n and k to small defaults for CI / quick runs.
    if args.smoke:
        if args.n == 100:   # only override if user didn't set --n explicitly
            args.n = SMOKE_N
        if args.k == 50:    # only override if user didn't set --k explicitly
            args.k = SMOKE_K

    out_json = Path(args.out).expanduser()
    out_md = Path(args.markdown).expanduser()
    if not out_json.is_absolute():
        out_json = REPO_ROOT / out_json
    if not out_md.is_absolute():
        out_md = REPO_ROOT / out_md

    from cinemateca.config import load_config

    print("Loading config + picking film…", flush=True)
    try:
        cfg = load_config()
    except Exception as exc:
        if args.smoke:
            print(
                f"::notice::bench_retrieval: config unavailable ({exc}); "
                "benchmark skipped (no-op pass).",
                flush=True,
            )
            return 0
        raise

    # Graceful no-index guard: if no CLIP index is on disk, skip instead of
    # crashing.  This is the normal CI path (runners have no Jeca Tatu data).
    if not _check_index_exists(cfg, film=args.film, smoke=args.smoke):
        msg = (
            f"No on-disk CLIP index found for film={args.film!r}"
            if args.film
            else "No on-disk CLIP index found for any registered film"
        )
        if args.smoke:
            print(
                f"::notice::bench_retrieval: {msg}; "
                "benchmark skipped (no-op pass — index absent in this environment).",
                flush=True,
            )
            print(
                f"SKIP: {msg}. "
                "Run `uv run cinemateca process <video>` to build an index, "
                "then re-run this benchmark.",
                flush=True,
            )
            return 0
        # Non-smoke: still a clean exit with a clear message rather than a
        # confusing traceback; _pick_film will produce a similar message but
        # this guard fires earlier and is more explicit.
        print(
            f"SKIP: {msg}. "
            "Run `uv run cinemateca process <video>` to build an index.",
            flush=True,
        )
        return 0

    slug = _pick_film(cfg, requested=args.film)

    print(f"Building fixture for film={slug!r}…", flush=True)
    fx = _build_fixture(cfg, slug=slug)
    print(
        f"  film={fx.slug}  scenes={fx.n_scenes}  vectors={fx.n_vectors}  "
        f"bm25_docs={fx.n_bm25_docs}  device={fx.device}",
        flush=True,
    )

    print("Building query pool…", flush=True)
    queries, prov = build_query_pool(
        n=args.n,
        queries_yaml=DEFAULT_QUERIES_YAML,
        metadata_dir=fx.metadata_dir,
    )
    print(
        f"  pool size={len(queries)} (yaml={prov['from_yaml']} "
        f"synthesised={prov['from_descriptions']} cycled={prov['cycled_repeats']})",
        flush=True,
    )

    results = run_bench(fx, queries, top_k=args.k)

    gpu = _gpu_info()
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "hardware": {
            "cpu_model": _cpu_model(),
            "cpu_count": _cpu_count(),
            "gpu": gpu,
            "platform": platform.platform(),
            **_torch_device_info(),
        },
        "fixture": {
            "slug": fx.slug,
            "n_scenes": fx.n_scenes,
            "n_vectors": fx.n_vectors,
            "n_bm25_docs": fx.n_bm25_docs,
            "device": fx.device,
            "min_similarity": fx.min_similarity,
        },
        "query_pool": prov,
        "config": {
            "sem_w": SEM_W,
            "bm25_w": BM25_W,
            "rrf_k": RRF_K,
            "warmup_queries": WARMUP_QUERIES,
        },
        "results": results,
    }

    write_json(payload, out_json)
    write_markdown(payload, out_md)

    # Echo headline so the caller sees it without opening files.
    hp95 = results["modes"]["hybrid"]["p95"]
    hp50 = results["modes"]["hybrid"]["p50"]
    hp99 = results["modes"]["hybrid"]["p99"]
    print("")
    print("─" * 70)
    print(
        f"Hybrid: p50={_fmt_ms(hp50)} ms  p95={_fmt_ms(hp95)} ms  p99={_fmt_ms(hp99)} ms  "
        f"(k={results['top_k']}, {fx.n_scenes} scenes, device={fx.device})"
    )
    print("─" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
