"""Global weighted-RRF fusion over per-film ranked lists (C1)."""

from __future__ import annotations

from collections.abc import Hashable
from typing import TypeVar

# The ranked-list key. Cross-film callers fuse on ``(slug, scene_id)``; the
# single-film path fuses on a bare ``scene_id``. The body only ever hashes the
# key, so one implementation serves both — it just has to say so to the typer.
K = TypeVar("K", bound=Hashable)


def fuse_global_rrf(
    weighted_lists: list[tuple[list[tuple[K, float]], float]],
    *,
    k_rrf: int,
) -> list[tuple[K, float]]:
    """Weighted RRF over >=2 globally ranked ``(key, score)`` lists.

    Each list contributes ``weight / (k_rrf + rank)`` per item; lists with
    ``weight <= 0`` are skipped. Returns items sorted by fused score, desc.
    """
    fused: dict[K, float] = {}
    for ranked, weight in weighted_lists:
        if weight <= 0.0:
            continue
        for rank, (key, _) in enumerate(ranked, start=1):
            fused[key] = fused.get(key, 0.0) + weight / (k_rrf + rank)
    return sorted(fused.items(), key=lambda pair: pair[1], reverse=True)
