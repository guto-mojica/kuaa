#!/usr/bin/env python3
"""End-to-end verification for audio search, CLIP×CLAP fusion, and the
cross-encoder reranker against real on-disk artifacts.

What it does:

  1. Audio search — runs three CLAP text queries against Jeca Tatu's
     ``data/library/jeca_tatu/audio/clap_embeddings.npy`` index, checks
     that hits come back with valid scene IDs, scores are descending, and
     that latency is in a reasonable budget.
  2. Fusion — runs cross-modal CLIP × CLAP linear-late-fusion at multiple
     ``visual_weight`` settings, checks the top-K changes meaningfully as
     ``w`` moves, and compares fusion top-5 vs CLIP-only top-5.
  3. Reranker — runs a hybrid baseline (top-50), then reruns through the
     cross-encoder ``rerank`` verb with a deterministic stub (default) or
     the real bge-reranker-v2-m3 (``--full``); checks the top-10 actually
     changes vs the baseline and that result cards stay intact.
  4. Graceful degradation — calls each dispatcher on the second film
     (Edwin Porter, no CLAP), confirms ``no_index=True`` returns (no 500),
     and that ``model='noop'`` short-circuits the reranker.

Exit codes:
  0 = every check passed
  1 = one or more checks failed (details printed to stderr).

Usage::

    uv run python scripts/verify_features.py              # default: stub cross-encoder
    uv run python scripts/verify_features.py --full       # use real bge-reranker-v2-m3

Default mode stubs the cross-encoder loader with a deterministic
description-length scoring function so we exercise the orchestrator
+ result-shape contract without paying the real bge-reranker-v2-m3
load (~1.1 GB). ``--full`` opts into the real model and downloads it
on first run if not already cached.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Project on PYTHONPATH the same way the bench script does it.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.WARNING, format="%(name)s %(levelname)s %(message)s")

# ─── Fixtures ─────────────────────────────────────────────────────────────────

JECA_SLUG = "jeca_tatu"
TRAIN_SLUG = "edwin_porter-the_great_train_robbery_1903"

# Audio queries — text designed to hit both modal extremes.
AUDIO_QUERIES = ["dialogue", "festive music", "footsteps on wood"]

# Fusion queries — one visual-heavy, one audio-heavy.
FUSION_QUERIES = [
    ("train", [0.3, 0.5, 0.7]),  # generic
    ("festive crowd music", [0.0, 0.5, 1.0]),  # CLAP-leaning + corner cases
]

# Reranker query — something the BM25/CLIP baseline can resolve and the
# cross-encoder can re-order based on descriptions.
RERANKER_QUERY = "homem a cavalo no campo"

LATENCY_BUDGET_MS = {
    "audio_per_query": 2000.0,  # CLAP encode + matmul on a film with ~400 scenes
    "fusion_per_query": 4000.0,  # CLIP + CLAP encode + merge
    "rerank_top_50": 30000.0,  # bge-reranker-v2-m3 over 50 short docs on CPU
}


# ─── Reporting ────────────────────────────────────────────────────────────────


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    latency_ms: float | None = None


@dataclass
class SectionResult:
    name: str
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)


def _emit_section(section: SectionResult) -> None:
    status = "PASS" if section.passed else "FAIL"
    print(f"\n=== {section.name}: {status} ===")
    for c in section.checks:
        mark = "ok  " if c.passed else "FAIL"
        lat = f" ({c.latency_ms:.1f} ms)" if c.latency_ms is not None else ""
        print(f"  [{mark}] {c.name}{lat}")
        if c.detail:
            for line in c.detail.splitlines():
                print(f"        {line}")


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _require_jeca() -> None:
    """Hard-fail fast if Jeca Tatu's CLIP+CLAP indices are missing."""
    clip = REPO_ROOT / "data/library" / JECA_SLUG / "embeddings/keyframe_embeddings.npy"
    clap = REPO_ROOT / "data/library" / JECA_SLUG / "audio/clap_embeddings.npy"
    missing = [p for p in (clip, clap) if not p.exists()]
    if missing:
        raise SystemExit(
            "Missing artefacts required for verification:\n  "
            + "\n  ".join(str(p) for p in missing)
        )


def _stub_reranker(monkey_target: Any) -> None:
    """Monkeypatch the cross-encoder loader with a deterministic stub.

    The stub rewards longer scene descriptions, which guarantees a
    non-trivial reorder vs the baseline CLIP/BM25 ranking on Jeca Tatu
    (where description length varies meaningfully across scenes). Used
    by the default verification mode so we exercise the orchestrator
    + result-shape contract without paying the ~1.1 GB bge load.
    """

    class _AnyStub:
        def compute_score(self, pairs: list[list[str]]) -> list[float]:
            return [float(len(d)) for _, d in pairs]

    # _load_reranker is wrapped in functools.lru_cache; clear it so the
    # stub doesn't get masked by a cached real model from a prior run.
    monkey_target._load_reranker.cache_clear()
    monkey_target._load_reranker = lambda _model_id: _AnyStub()


# ─── Verifications ────────────────────────────────────────────────────────────


def verify_audio_search(cfg: Any) -> SectionResult:
    """Audio-only CLAP search on Jeca Tatu."""
    from api.services.search import dispatch_audio_search
    from cinemateca.library import FilmContext

    section = SectionResult(name="Audio-only search (Jeca Tatu, real CLAP index)")
    ctx = FilmContext.for_film(cfg, JECA_SLUG)

    # Warm-up: first invocation pays the CLAP model load (~3–4 s on CPU,
    # ~1 s on GPU). Time the steady-state queries instead.
    print("  warming CLAP embedder...", end="", flush=True)
    t_warm = time.perf_counter()
    dispatch_audio_search(cfg, ctx, "warmup", top_k=1)
    print(f" done ({(time.perf_counter() - t_warm) * 1000.0:.0f} ms)")

    for q in AUDIO_QUERIES:
        t0 = time.perf_counter()
        try:
            hits, no_index = dispatch_audio_search(cfg, ctx, q, top_k=5)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            ok = (
                not no_index
                and len(hits) == 5
                and all(isinstance(h["scene_id"], int) and h["scene_id"] >= 0 for h in hits)
                and [h["score"] for h in hits] == sorted([h["score"] for h in hits], reverse=True)
                and lat_ms < LATENCY_BUDGET_MS["audio_per_query"]
            )
            detail = (
                f"top scene_ids={[h['scene_id'] for h in hits]} "
                f"scores={[round(h['score'], 3) for h in hits]}"
            )
            if not ok:
                if no_index:
                    detail += " | no_index=True UNEXPECTED"
                if len(hits) != 5:
                    detail += f" | got {len(hits)} hits, want 5"
                if lat_ms >= LATENCY_BUDGET_MS["audio_per_query"]:
                    detail += (
                        f" | latency {lat_ms:.0f} ms exceeds budget "
                        f"{LATENCY_BUDGET_MS['audio_per_query']:.0f} ms"
                    )
            section.checks.append(
                CheckResult(name=f"query={q!r}", passed=ok, detail=detail, latency_ms=lat_ms)
            )
        except Exception as exc:
            section.checks.append(
                CheckResult(
                    name=f"query={q!r}",
                    passed=False,
                    detail=f"EXCEPTION: {type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                )
            )
    return section


def verify_fusion(cfg: Any) -> SectionResult:
    """Linear-late-fusion CLIP × CLAP on Jeca Tatu, multiple w settings."""
    from api.services.search import dispatch_fusion_search
    from cinemateca.library import FilmContext

    section = SectionResult(name="CLIP × CLAP fusion (Jeca Tatu, both indices)")
    ctx = FilmContext.for_film(cfg, JECA_SLUG)

    # Warm-up: first invocation pays both CLIP and CLAP model loads. Time
    # the steady-state queries instead.
    print("  warming CLIP+CLAP embedders...", end="", flush=True)
    t_warm = time.perf_counter()
    dispatch_fusion_search(cfg, ctx, "warmup", top_k=1, visual_weight=0.5)
    print(f" done ({(time.perf_counter() - t_warm) * 1000.0:.0f} ms)")

    # Track top-5s per query for cross-w comparison.
    per_query_results: dict[str, dict[float, list[int]]] = {}
    for q, weights in FUSION_QUERIES:
        per_query_results[q] = {}
        for w in weights:
            t0 = time.perf_counter()
            try:
                hits, no_index = dispatch_fusion_search(cfg, ctx, q, top_k=5, visual_weight=w)
                lat_ms = (time.perf_counter() - t0) * 1000.0
                ok = (
                    not no_index
                    and len(hits) == 5
                    and all(isinstance(h["scene_id"], int) for h in hits)
                    and "clip_score" in hits[0]
                    and "clap_score" in hits[0]
                    and lat_ms < LATENCY_BUDGET_MS["fusion_per_query"]
                )
                detail = (
                    f"scene_ids={[h['scene_id'] for h in hits]} "
                    f"scores={[round(h['score'], 3) for h in hits]}"
                )
                if not ok:
                    if no_index:
                        detail += " | no_index=True UNEXPECTED"
                    if len(hits) != 5:
                        detail += f" | got {len(hits)} hits, want 5"
                    if lat_ms >= LATENCY_BUDGET_MS["fusion_per_query"]:
                        detail += (
                            f" | latency {lat_ms:.0f} ms exceeds budget "
                            f"{LATENCY_BUDGET_MS['fusion_per_query']:.0f} ms"
                        )
                per_query_results[q][w] = [h["scene_id"] for h in hits]
                section.checks.append(
                    CheckResult(
                        name=f"query={q!r} w={w}",
                        passed=ok,
                        detail=detail,
                        latency_ms=lat_ms,
                    )
                )
            except Exception as exc:
                section.checks.append(
                    CheckResult(
                        name=f"query={q!r} w={w}",
                        passed=False,
                        detail=f"EXCEPTION: {type(exc).__name__}: {exc}",
                    )
                )

    # Cross-w differentiation: a meaningful fusion should change top-5 as w
    # moves from 0.0 (pure CLAP) to 1.0 (pure CLIP). Compare those two.
    for q, weight_lists in per_query_results.items():
        if 0.0 in weight_lists and 1.0 in weight_lists:
            clap_top = weight_lists[0.0]
            clip_top = weight_lists[1.0]
            differs = clap_top != clip_top
            section.checks.append(
                CheckResult(
                    name=f"query={q!r} top-5 differs between w=0.0 (CLAP-only) and w=1.0 (CLIP-only)",
                    passed=differs,
                    detail=f"CLAP-only={clap_top} | CLIP-only={clip_top}",
                )
            )
    return section


def verify_reranker(cfg: Any, fast: bool) -> SectionResult:
    """Hybrid baseline top-50 → rerank top-10. Verify the order shifts."""
    import sys as _sys

    from api.services.search import dispatch_text_search, rerank_template_results
    from cinemateca.library import FilmContext

    section = SectionResult(name="Cross-encoder reranker (hybrid baseline → rerank)")
    ctx = FilmContext.for_film(cfg, JECA_SLUG)

    if fast:
        import cinemateca.search  # noqa: F401

        rerank_mod = _sys.modules["cinemateca.search.rerank"]
        _stub_reranker(rerank_mod)

    # Baseline: hybrid top-50.
    try:
        # Pull hybrid weights from cfg the same way the route does.
        sw = float(cfg.search.hybrid_sem_w)
        bw = float(cfg.search.hybrid_bm25_w)
        rrf_k = int(cfg.search.bm25.rrf_k)
        result_df, no_index = dispatch_text_search(
            cfg, ctx, RERANKER_QUERY, [], 50, 0.0, "hybrid", sw, bw, rrf_k
        )
        if no_index or result_df is None:
            section.checks.append(
                CheckResult(
                    name="baseline hybrid top-50",
                    passed=False,
                    detail="dispatch_text_search returned no_index=True",
                )
            )
            return section

        # Convert per-film DataFrame → enriched dict-rows the route uses.
        from api.routes.search import _enriched_per_film

        baseline_rows = _enriched_per_film(cfg, ctx, result_df, JECA_SLUG)
        baseline_ids = [int(r["scene_id"]) for r in baseline_rows]
        has_desc = sum(1 for r in baseline_rows if r.get("description"))

        section.checks.append(
            CheckResult(
                name=f"baseline hybrid produced 50 rows, {has_desc} with description",
                passed=len(baseline_rows) >= 10 and has_desc >= 5,
                detail=f"len(rows)={len(baseline_rows)} top-10={baseline_ids[:10]}",
            )
        )
    except Exception as exc:
        section.checks.append(
            CheckResult(
                name="baseline hybrid",
                passed=False,
                detail=f"EXCEPTION: {type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            )
        )
        return section

    # Reranker: explicit enable.
    try:
        t0 = time.perf_counter()
        reranked = rerank_template_results(
            baseline_rows,
            cfg=cfg,
            query=RERANKER_QUERY,
            mode="hybrid",
            enabled=True,
        )
        lat_ms = (time.perf_counter() - t0) * 1000.0
        top10_reranked = [int(r["scene_id"]) for r in reranked[:10]]
        top10_baseline = baseline_ids[:10]
        order_changed = top10_reranked != top10_baseline
        any_rerank_score = any(r.get("rerank_score") is not None for r in reranked[:10])
        cards_intact = all(
            isinstance(r.get("scene_id"), int) and r.get("similarity") is not None
            for r in reranked[:10]
        )
        budget_ok = lat_ms < LATENCY_BUDGET_MS["rerank_top_50"]
        ok = order_changed and any_rerank_score and cards_intact and budget_ok
        detail = (
            f"top-10 baseline={top10_baseline}\n"
            f"top-10 reranked={top10_reranked}\n"
            f"any rerank_score={any_rerank_score} cards_intact={cards_intact} "
            f"latency={lat_ms:.0f}ms budget={LATENCY_BUDGET_MS['rerank_top_50']:.0f}ms"
        )
        section.checks.append(
            CheckResult(
                name="rerank actually fires (top-10 order changes vs hybrid)",
                passed=ok,
                detail=detail,
                latency_ms=lat_ms,
            )
        )
    except Exception as exc:
        section.checks.append(
            CheckResult(
                name="rerank fires",
                passed=False,
                detail=f"EXCEPTION: {type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            )
        )

    return section


def verify_graceful_degradation(cfg: Any) -> SectionResult:
    """No-index empty states + reranker noop short-circuit + fusion w/ only CLIP."""
    from api.services.search import (
        apply_reranker,
        dispatch_audio_search,
        dispatch_fusion_search,
    )
    from cinemateca.library import FilmContext
    from cinemateca.search.types import Hit, Query, SearchResult

    section = SectionResult(name="Graceful degradation")

    # 1. Audio on Train Robbery (no CLAP).
    try:
        ctx_train = FilmContext.for_film(cfg, TRAIN_SLUG)
        hits, no_index = dispatch_audio_search(cfg, ctx_train, "anything", top_k=5)
        ok = no_index is True and hits == []
        section.checks.append(
            CheckResult(
                name="audio on film without CLAP → no_index=True (Edwin Porter)",
                passed=ok,
                detail=f"no_index={no_index} len(hits)={len(hits)}",
            )
        )
    except Exception as exc:
        section.checks.append(
            CheckResult(
                name="audio on film without CLAP",
                passed=False,
                detail=f"EXCEPTION (should have returned no_index, not raised): "
                f"{type(exc).__name__}: {exc}",
            )
        )

    # 2. Fusion on Train Robbery (only CLIP, no CLAP) — should DEGRADE to
    # CLIP-only, not crash. The verb's contract is "missing modalities
    # don't actively penalise", so we expect hits to come back, not no_index.
    try:
        ctx_train = FilmContext.for_film(cfg, TRAIN_SLUG)
        hits, no_index = dispatch_fusion_search(
            cfg, ctx_train, "train robbery", top_k=5, visual_weight=0.5
        )
        # Train Robbery has CLIP — fusion should still return CLIP-only results
        # at w*clip score (CLAP side contributes 0 because the modality is absent).
        ok = (not no_index) and len(hits) > 0 and all(h.get("clap_score") == 0.0 for h in hits)
        section.checks.append(
            CheckResult(
                name="fusion on CLIP-only film degrades to CLIP-only (Edwin Porter)",
                passed=ok,
                detail=(
                    f"no_index={no_index} len(hits)={len(hits)} "
                    f"clap_scores={[h.get('clap_score') for h in hits[:3]]}"
                ),
            )
        )
    except Exception as exc:
        section.checks.append(
            CheckResult(
                name="fusion on CLIP-only film",
                passed=False,
                detail=f"EXCEPTION: {type(exc).__name__}: {exc}",
            )
        )

    # 3. Reranker noop escape hatch — exercise via apply_reranker.
    try:
        synthetic = SearchResult(
            hits=[
                Hit(scene_id=1, score=0.9, keyframe_path="", description="d1"),
                Hit(scene_id=2, score=0.5, keyframe_path="", description="d2"),
            ],
            mode="hybrid",
            weights=None,
            query=Query.text_query("q"),
        )
        from types import SimpleNamespace

        cfg_noop = SimpleNamespace(
            retrieval=SimpleNamespace(
                reranker=SimpleNamespace(enabled=True, model="noop", top_k_in=20),
            )
        )
        out = apply_reranker(synthetic, cfg=cfg_noop)
        ok = out.hits == synthetic.hits
        section.checks.append(
            CheckResult(
                name="reranker model='noop' passthrough leaves hits unchanged",
                passed=ok,
                detail=(
                    f"in_ids={[h.scene_id for h in synthetic.hits]} "
                    f"out_ids={[h.scene_id for h in out.hits]}"
                ),
            )
        )
    except Exception as exc:
        section.checks.append(
            CheckResult(
                name="reranker noop",
                passed=False,
                detail=f"EXCEPTION: {type(exc).__name__}: {exc}",
            )
        )

    return section


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use the real BAAI/bge-reranker-v2-m3 (downloads ~1.1 GB on first run).",
    )
    args = parser.parse_args()
    fast = not args.full

    _require_jeca()

    # Lazy import after sys.path setup.
    from api.deps import get_config

    cfg = get_config()
    sections: list[SectionResult] = []

    print("Verifying audio / fusion / reranker against real artifacts...")
    print(f"  library_dir: {cfg.paths.library_dir}")
    print(
        f"  reranker mode: {'real BAAI/bge-reranker-v2-m3' if not fast else 'stubbed (use --full for real model)'}"
    )

    sections.append(verify_audio_search(cfg))
    sections.append(verify_fusion(cfg))
    sections.append(verify_reranker(cfg, fast=fast))
    sections.append(verify_graceful_degradation(cfg))

    for s in sections:
        _emit_section(s)

    overall = all(s.passed for s in sections)
    print()
    if overall:
        print("RESULT: all 4 feature areas verified end-to-end.")
        return 0
    failed_areas = [s.name for s in sections if not s.passed]
    print("RESULT: FAILURES in:")
    for n in failed_areas:
        print(f"  - {n}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
