#!/usr/bin/env python3
"""End-to-end verification for the cross-encoder reranker against real
on-disk artifacts.

What it does:

  1. Reranker — runs a hybrid baseline (top-50), then reruns through the
     cross-encoder ``rerank`` verb with a deterministic stub (default) or
     the real bge-reranker-v2-m3 (``--full``); checks the top-10 actually
     changes vs the baseline and that result cards stay intact.
  2. Graceful degradation — confirms ``model='noop'`` short-circuits the
     reranker (passthrough leaves hits unchanged).

Exit codes:
  0 = every check passed
  1 = one or more checks failed (details printed to stderr).

Usage::

    uv run python scripts/verify_reranker.py              # default: stub cross-encoder
    uv run python scripts/verify_reranker.py --full       # use real bge-reranker-v2-m3

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

# Reranker query — something the BM25/CLIP baseline can resolve and the
# cross-encoder can re-order based on descriptions.
RERANKER_QUERY = "homem a cavalo no campo"

LATENCY_BUDGET_MS = {
    # bge-reranker-v2-m3 over 50 short docs on CPU. The stub mode used by
    # default completes in <1 ms; the budget is sized for the full model.
    "rerank_top_50": 30000.0,
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
    """Hard-fail fast if Jeca Tatu's CLIP index is missing."""
    clip = REPO_ROOT / "data/library" / JECA_SLUG / "embeddings/keyframe_embeddings.npy"
    if not clip.exists():
        raise SystemExit("Missing artefacts required for verification:\n  " + str(clip))


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


def verify_reranker(cfg: Any, fast: bool) -> SectionResult:
    """Hybrid baseline top-50 → rerank top-10. Verify the order shifts."""
    import sys as _sys

    from api.services.search import (
        cards_to_result,
        dispatch_text_search,
        rerank_search_result,
        result_to_cards,
    )
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
        result, originals = cards_to_result(baseline_rows, query=RERANKER_QUERY, mode="hybrid")
        result = rerank_search_result(result, cfg=cfg, enabled=True)
        reranked = result_to_cards(result, originals)
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
    """Reranker noop escape-hatch short-circuit."""
    from api.services.search import apply_reranker
    from cinemateca.search.types import Hit, Query, SearchResult

    section = SectionResult(name="Graceful degradation")

    # Reranker noop escape hatch — exercise via apply_reranker.
    try:
        synthetic = SearchResult(
            hits=[
                Hit(scene_id=1, score=0.9, keyframe_path="", description="d1"),
                Hit(scene_id=2, score=0.5, keyframe_path="", description="d2"),
            ],
            mode="hybrid",
            weights=None,
            query=Query.of_text("q"),
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

    print("Verifying the cross-encoder reranker against real artifacts...")
    print(f"  library_dir: {cfg.paths.library_dir}")
    print(
        f"  reranker mode: {'real BAAI/bge-reranker-v2-m3' if not fast else 'stubbed (use --full for real model)'}"
    )

    sections.append(verify_reranker(cfg, fast=fast))
    sections.append(verify_graceful_degradation(cfg))

    for s in sections:
        _emit_section(s)

    overall = all(s.passed for s in sections)
    print()
    if overall:
        print("RESULT: reranker verified end-to-end.")
        return 0
    failed_areas = [s.name for s in sections if not s.passed]
    print("RESULT: FAILURES in:")
    for n in failed_areas:
        print(f"  - {n}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
