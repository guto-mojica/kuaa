#!/usr/bin/env python3
"""Retrieval-latency benchmark for the hybrid search pipeline.

Times the three retriever modes (clip / bm25 / hybrid) against one
already-indexed film and reports p50 / p95 / p99 / mean / max per mode.
For ``hybrid`` it additionally breaks the per-query latency into the four
sub-stages that the dispatcher actually executes:

  1. ``clip_encode``  — ``OpenClipEmbedder.encode_text(query)``
  2. ``clip_search``  — ``embeddings @ vec`` + scene-id dedup + top-K
  3. ``bm25_query``   — ``BM25Index.query(query, top_k=raw_k)``
  4. ``rrf_fuse``     — ``fuse_rrf(clip_ranked, bm25_hits, ...)``

These are the same four operations the production ``search_hybrid``
dispatcher calls (``api/services/search.py``); the breakdown is a
strictly additive instrumentation, no behaviour change.

Usage::

    uv run python scripts/bench_retrieval.py
    uv run python scripts/bench_retrieval.py --n 200 --k 50 --film jeca_tatu
    uv run python scripts/bench_retrieval.py --out data/perf/run_2026-05-24.json

Outputs both a JSON results file (defaults to ``data/perf/bench_results.json``)
and a Markdown summary (``docs/PERFORMANCE.md``). The Markdown file
captures hardware, headline number, and ready-to-paste README snippet —
the JSON file is the raw record (per-mode + per-stage stats + every
sample, for re-analysis).

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
    embedder: object  # cinemateca.models.clip.openclip.OpenClipEmbedder
    index: object  # api.services.search.SearchIndex
    bm25: object  # cinemateca.retrieval.bm25.BM25Index
    device: str
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
    from cinemateca.models.clip.openclip import OpenClipEmbedder

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

    embedder = OpenClipEmbedder(cfg, device)
    # Force the lazy-load so warm-up is the only timing-sensitive cost left.
    embedder._load_model()

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
        library_dir=Path(cfg.paths.library_dir),
        metadata_dir=ctx.metadata_dir,
    )


# ─── Stage primitives ─────────────────────────────────────────────────────────
#
# Each helper does exactly the work the production dispatcher does for
# that stage, so the sub-stage breakdown sums to (approximately) the
# whole-query latency on the hybrid path. They are intentionally
# minimal — no logging, no error handling beyond what the production
# path already handles — so we measure code, not instrumentation overhead.


def _clip_encode(embedder, query: str):
    """One CLIP text encode → unit-norm vector (no batching, mirrors prod)."""
    import numpy as np

    vec = embedder.encode_text(query)
    norm = float(np.linalg.norm(vec))
    return vec / (norm + 1e-12)


def _clip_search_fast(index, vec, top_k: int) -> list[tuple[int, float]]:
    """np.argsort + top-K + scene-id dedup — matches production
    ``embeddings.SemanticSearch.by_text`` (the per-film CLIP path).

    Returns ``[(scene_id, cosine), …]`` of length ≤ top_k after dedup.
    This is what ``search_text`` runs in ``retriever=clip`` mode.
    """
    import numpy as np

    scores: np.ndarray = (index.embeddings @ vec).flatten()  # type: ignore[operator]
    # Widen by 4× to mirror search_text's keyframe-density widening —
    # the post-dedup top-K still needs ``top_k`` distinct scenes.
    raw = max(top_k * 4, 1)
    top_idx = np.argsort(scores)[::-1][:raw]
    sids_col = index.kf_df["scene_id"].to_numpy()
    seen: set[int] = set()
    out: list[tuple[int, float]] = []
    for i in top_idx:
        sid = int(sids_col[i])
        if sid in seen:
            continue
        seen.add(sid)
        out.append((sid, float(scores[i])))
        if len(out) >= top_k:
            break
    return out


def _clip_search_best_per_sid(index, vec, top_k: int) -> list[tuple[int, float]]:
    """Full-iteration best-cosine-per-scene_id pass — matches
    ``aggregate_search`` (lines 678-694) and
    ``_best_row_by_sid_from_embeddings`` used by the hybrid dispatcher.

    More expensive than ``_clip_search_fast`` because it touches every
    row of ``kf_df`` (needed for the BM25-only backfill: a scene that
    surfaces via BM25 alone must still find ITS best CLIP keyframe to
    display). We use this for the ``clip_search`` sub-stage timing so
    the breakdown reflects what the hybrid path actually pays.
    """
    import numpy as np

    scores: np.ndarray = (index.embeddings @ vec).flatten()  # type: ignore[operator]
    sids_col = index.kf_df["scene_id"].to_numpy()
    best: dict[int, float] = {}
    for i in range(scores.shape[0]):
        sid = int(sids_col[i])
        s = float(scores[i])
        prev = best.get(sid)
        if prev is None or s > prev:
            best[sid] = s
    ranked = sorted(best.items(), key=lambda p: p[1], reverse=True)[:top_k]
    return ranked


def _bm25_query(bm25, query: str, top_k: int) -> list[tuple[int, float]]:
    """One BM25 query → ranked list (delegates straight to the index)."""
    return bm25.query(query, top_k=top_k)


def _rrf_fuse(clip_ranked, bm25_hits, top_k: int) -> list[tuple[int, float]]:
    from cinemateca.retrieval.hybrid import fuse_rrf

    return fuse_rrf(clip_ranked, bm25_hits, sem_w=SEM_W, bm25_w=BM25_W, k_rrf=RRF_K)[:top_k]


# ─── Per-query timed measurements ─────────────────────────────────────────────


def _time_clip(fx: BenchFixture, query: str, *, raw_k: int, top_k: int) -> float:
    """Total ms for a CLIP-only query (encode + matmul + argsort + dedup + top_k).

    Uses the fast argsort/topK path that production runs in
    ``retriever=clip`` mode (``SemanticSearch.by_text`` +
    ``search_text``'s scene-id dedup), NOT the full-iteration
    best-per-sid pass the hybrid path needs.
    """
    t0 = time.perf_counter()
    vec = _clip_encode(fx.embedder, query)
    _ = _clip_search_fast(fx.index, vec, top_k)
    return (time.perf_counter() - t0) * 1000.0


def _time_bm25(fx: BenchFixture, query: str, *, raw_k: int, top_k: int) -> float:
    """Total ms for a BM25-only query."""
    t0 = time.perf_counter()
    _ = _bm25_query(fx.bm25, query, raw_k)
    return (time.perf_counter() - t0) * 1000.0


def _time_hybrid(fx: BenchFixture, query: str, *, raw_k: int, top_k: int) -> dict:
    """Total + 4 sub-stage timings for one hybrid query.

    Returns ``{"total": ms, "clip_encode": ms, "clip_search": ms,
    "bm25_query": ms, "rrf_fuse": ms}``. The four sub-stage sums should
    be ≤ total (small difference = ``time.perf_counter()`` jitter +
    glue cost).
    """
    t0 = time.perf_counter()

    t_a = time.perf_counter()
    vec = _clip_encode(fx.embedder, query)
    t_b = time.perf_counter()
    # Hybrid uses the full-iteration best-per-sid pass (matches
    # production: search_text -> by_text returns its top-K then
    # _best_row_by_sid_from_embeddings walks the full matrix for
    # BM25-backfill display purposes).
    clip_ranked = _clip_search_best_per_sid(fx.index, vec, raw_k)
    t_c = time.perf_counter()
    bm25_hits = _bm25_query(fx.bm25, query, raw_k)
    t_d = time.perf_counter()
    _ = _rrf_fuse(clip_ranked, bm25_hits, top_k)
    t_e = time.perf_counter()

    total = (t_e - t0) * 1000.0
    return {
        "total": total,
        "clip_encode": (t_b - t_a) * 1000.0,
        "clip_search": (t_c - t_b) * 1000.0,
        "bm25_query": (t_d - t_c) * 1000.0,
        "rrf_fuse": (t_e - t_d) * 1000.0,
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
    # Mirrors search_text's 4× widening so the dedup pass has room to
    # surface ``top_k`` distinct scenes.
    raw_k = max(top_k * 4, 1)

    print(f"  warm-up: {WARMUP_QUERIES} throwaway queries…", flush=True)
    _warmup(fx, raw_k=raw_k, top_k=top_k)

    print(f"  timing {len(queries)} queries x 3 modes (k={top_k}, raw_k={raw_k})…", flush=True)
    clip_ms: list[float] = []
    bm25_ms: list[float] = []
    hybrid_total_ms: list[float] = []
    stage_ms: dict[str, list[float]] = {
        "clip_encode": [],
        "clip_search": [],
        "bm25_query": [],
        "rrf_fuse": [],
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
            "hybrid_clip_encode": stage_ms["clip_encode"],
            "hybrid_clip_search": stage_ms["clip_search"],
            "hybrid_bm25_query": stage_ms["bm25_query"],
            "hybrid_rrf_fuse": stage_ms["rrf_fuse"],
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
        f"dispatcher trims to `k={top_k}`."
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
    lines.append("| Stage         | p50  | p95  | p99  | mean | max  |")
    lines.append("|---------------|-----:|-----:|-----:|-----:|-----:|")
    for stage in ("clip_encode", "clip_search", "bm25_query", "rrf_fuse"):
        s = stages[stage]
        lines.append(
            f"| {stage:<13} | {_fmt_ms(s['p50']):>4} | {_fmt_ms(s['p95']):>4} | "
            f"{_fmt_ms(s['p99']):>4} | {_fmt_ms(s['mean']):>4} | {_fmt_ms(s['max']):>4} |"
        )
    lines.append("")
    lines.append(
        "The four sub-stages run sequentially inside the dispatcher "
        "(`api/services/search.py::search_hybrid`). Adding them ≈ recovers "
        "the hybrid total above; the small gap is glue cost + "
        "`time.perf_counter()` jitter."
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_json = Path(args.out).expanduser()
    out_md = Path(args.markdown).expanduser()
    if not out_json.is_absolute():
        out_json = REPO_ROOT / out_json
    if not out_md.is_absolute():
        out_md = REPO_ROOT / out_md

    from cinemateca.config import load_config

    print("Loading config + picking film…", flush=True)
    cfg = load_config()
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
