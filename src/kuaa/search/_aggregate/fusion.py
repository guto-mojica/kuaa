"""Global weighted-RRF fusion over per-film ranked lists (C1)."""

from __future__ import annotations


def fuse_global_rrf(
    weighted_lists: list[tuple[list[tuple[tuple[str, int], float]], float]],
    *,
    k_rrf: int,
) -> list[tuple[tuple[str, int], float]]:
    """Weighted RRF over >=2 globally ranked ``((slug, scene_id), score)`` lists.

    Each list contributes ``weight / (k_rrf + rank)`` per item; lists with
    ``weight <= 0`` are skipped. Returns items sorted by fused score, desc.
    """
    fused: dict[tuple[str, int], float] = {}
    for ranked, weight in weighted_lists:
        if weight <= 0.0:
            continue
        for rank, (key, _) in enumerate(ranked, start=1):
            fused[key] = fused.get(key, 0.0) + weight / (k_rrf + rank)
    return sorted(fused.items(), key=lambda pair: pair[1], reverse=True)
