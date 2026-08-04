"""Weighted Reciprocal Rank Fusion (RRF) + retriever-mode resolution.

Pure-functional. Takes ranked lists in, returns a fused ranked list out.
Does not know about CLIP, BM25, or any specific retriever — those are
the caller's responsibility (the service layer).

Why RRF-with-weights (not weighted-linear over normalised scores)?
RRF works on ranks, which are scale-invariant — no need to calibrate
CLIP cosines against BM25 scores. The Cormack et al. 2009 paper is the
canonical reference; the master spec commits to RRF.
"""

from __future__ import annotations

from collections.abc import Iterable

# Public so tests can pin the constant.
DEFAULT_RRF_K: int = 60

# Majority share of the exact-lexical metadata list in 3-way hybrid fusion
# (tags / descriptions / detected objects vs the CLIP+BM25 residual). The
# single source of truth — ``cfg.search.hybrid_metadata_w`` defaults to it,
# and ``search_hybrid`` / ``aggregate._dispatch_ranked`` / the eval harness
# all resolve through it.
DEFAULT_METADATA_W: float = 0.65


def resolve_metadata_w(cfg: object | None) -> float:
    """Resolve the metadata fusion share from ``cfg.search.hybrid_metadata_w``.

    Duck-typed so unit-test configs without a ``search`` section (and callers
    with ``cfg=None``) fall back to :data:`DEFAULT_METADATA_W`. Clamped to
    ``[0, 1]``; ``0.0`` disables the metadata leg entirely.
    """
    search = getattr(cfg, "search", None)
    raw = getattr(search, "hybrid_metadata_w", DEFAULT_METADATA_W)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_METADATA_W
    return min(max(value, 0.0), 1.0)


def fuse_rrf(
    list_a: Iterable[tuple[int, float]],
    list_b: Iterable[tuple[int, float]],
    *,
    sem_w: float,
    bm25_w: float,
    k_rrf: int = DEFAULT_RRF_K,
) -> list[tuple[int, float]]:
    """Fuse two ranked lists by weighted RRF.

    Args:
        list_a: ``[(scene_id, score), …]`` ranked descending by score.
            Treated as the "semantic" / CLIP side.
        list_b: same shape, treated as the BM25 side.
        sem_w: weight applied to list_a's RRF contribution.
        bm25_w: weight applied to list_b's contribution.
        k_rrf: rank-shift constant. Cormack et al. used 60.

    Returns:
        Fused ``[(scene_id, fused_score), …]`` sorted descending. Every
        scene_id that appears in either input is in the output.

    Precondition:
        Each input list must already be deduped by ``scene_id``. Repeated
        entries within a list cause the rank-by-sid dict to silently
        retain only the last occurrence, which is rarely what callers
        want. Today's callers (CLIP search + BM25Index.query) both dedupe
        before returning.
    """
    ranks_a: dict[int, int] = {sid: rank for rank, (sid, _) in enumerate(list_a, start=1)}
    ranks_b: dict[int, int] = {sid: rank for rank, (sid, _) in enumerate(list_b, start=1)}
    all_sids = set(ranks_a) | set(ranks_b)
    fused: list[tuple[int, float]] = []
    for sid in all_sids:
        score = 0.0
        if sid in ranks_a:
            score += sem_w / (k_rrf + ranks_a[sid])
        if sid in ranks_b:
            score += bm25_w / (k_rrf + ranks_b[sid])
        fused.append((sid, score))
    fused.sort(key=lambda pair: pair[1], reverse=True)
    return fused


def resolve_weights(
    *, sem_w: float, bm25_w: float, defaults: tuple[float, float]
) -> tuple[float, float]:
    """Clamp weights into ``[0, 1]`` and fall back on the degenerate case.

    ``(0, 0)`` would make every fused score zero — ordering becomes
    undefined. Fall back to the configured defaults instead of silently
    sorting by some incidental tie-break.
    """
    sw = max(0.0, min(1.0, float(sem_w)))
    bw = max(0.0, min(1.0, float(bm25_w)))
    if sw == 0.0 and bw == 0.0:
        return defaults
    return sw, bw
