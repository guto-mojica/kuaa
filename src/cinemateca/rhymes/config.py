"""Read cfg.rhymes tunables (top_k, lambda, min_similarity)."""

from __future__ import annotations

from cinemateca.config import Settings


def rimas_cfg(cfg: Settings) -> tuple[int, float, float]:
    """Read ``cfg.rimas.{top_n,mmr_lambda,threshold}`` with sensible defaults.

    Test configs built off a minimal SimpleNamespace may omit the
    ``rimas`` section entirely. The defaults here mirror
    ``config/default.yaml`` so a missing config never collapses the
    page.
    """
    rimas = getattr(cfg, "rimas", None)
    top_n = int(getattr(rimas, "top_n", 8))
    mmr_lambda = float(getattr(rimas, "mmr_lambda", 0.5))
    threshold = float(getattr(rimas, "threshold", 0.75))
    return top_n, mmr_lambda, threshold
