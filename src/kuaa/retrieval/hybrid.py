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

# Share of the exact-lexical metadata list in 3-way hybrid fusion (tags /
# descriptions / detected objects vs the CLIP+BM25 residual). The single
# source of truth — ``cfg.search.hybrid_metadata_w`` defaults to it, and
# ``search_hybrid`` / ``aggregate._dispatch_ranked`` / the eval harness all
# resolve through it.
#
# This is the ceiling, applied to *short* queries. See
# :func:`effective_metadata_w` for why it tapers.
DEFAULT_METADATA_W: float = 0.65

# The metadata share for long natural-language queries. Derived from a sweep
# on ``m3_text_queries`` (15 queries, 4-7 terms each) once the leg actually
# started firing: nDCG@10 ran 0.073 at w<=0.10, 0.067 at 0.20-0.35, and 0.042
# at 0.50-0.65 — monotonically against the leg.
LONG_QUERY_METADATA_W: float = 0.15

# Query lengths (in scored terms) that bracket the taper.
_SHORT_QUERY_TERMS: int = 2
_LONG_QUERY_TERMS: int = 5


def effective_metadata_w(base_w: float, query_terms: int) -> float:
    """Scale the metadata fusion share by how long the query is.

    The exact-lexical leg exists to rescue *short object queries* — the
    scorer's own docstring says so: for ``dog``, an exact tag or detector
    class is stronger evidence than a weak visual cosine. At that length it
    earns a majority share, and a synthetic worst case (CLIP ranking 58
    wrong scenes above the two tagged ones) shows it genuinely needs one.

    On long natural-language queries the same weight is actively harmful,
    because a partial lexical overlap is weak evidence that nonetheless
    outranks a good dense match. The original code encoded this intuition
    as a hard ``len(query_tokens) > 4 -> return {}`` bail-out, which threw
    the leg away entirely rather than turning it down — and, because the
    leg was then inert on almost every real query, left its 0.65 weight
    unmeasured for as long as it shipped.

    Tapering keeps both properties: full strength where the evidence is
    strong, :data:`LONG_QUERY_METADATA_W` where it is not.

    Args:
        base_w: the configured ceiling (``cfg.search.hybrid_metadata_w``).
        query_terms: number of scored query terms.

    Returns:
        The share to use, never above ``base_w``.
    """
    floor = min(base_w, LONG_QUERY_METADATA_W)
    if query_terms <= _SHORT_QUERY_TERMS:
        return base_w
    if query_terms >= _LONG_QUERY_TERMS:
        return floor
    span = (query_terms - _SHORT_QUERY_TERMS) / (_LONG_QUERY_TERMS - _SHORT_QUERY_TERMS)
    return base_w + (floor - base_w) * span


def resolve_metadata_w(cfg: object | None, query: str | None = None) -> float:
    """Resolve the metadata fusion share from ``cfg.search.hybrid_metadata_w``.

    Duck-typed so unit-test configs without a ``search`` section (and callers
    with ``cfg=None``) fall back to :data:`DEFAULT_METADATA_W`. Clamped to
    ``[0, 1]``; ``0.0`` disables the metadata leg entirely.

    When ``query`` is supplied the configured value is treated as a ceiling
    and tapered by query length — see :func:`effective_metadata_w`. Callers
    that omit it get the untapered ceiling, which is the right default for
    a caller that has no query in hand.
    """
    search = getattr(cfg, "search", None)
    raw = getattr(search, "hybrid_metadata_w", DEFAULT_METADATA_W)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = DEFAULT_METADATA_W
    base = min(max(value, 0.0), 1.0)
    if query is None:
        return base
    # Imported here: kuaa.search depends on kuaa.retrieval, not the reverse.
    from kuaa.search._aggregate.scorers import build_query_terms

    return effective_metadata_w(base, len(build_query_terms(query)))


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
    # Accumulate into an insertion-ordered dict rather than iterating a set.
    # A set loses insertion order, so equal fused scores used to break on
    # int-hash bucket order — and with k_rrf=60 near-ties are common (ranks
    # 1 and 2 differ by only 1.6%), making the top of the result list
    # unstable for reasons unrelated to relevance. Seeding from list_a then
    # list_b makes ties break by first-seen leg, matching fuse_global_rrf.
    scores: dict[int, float] = {}
    for sid in (*ranks_a, *ranks_b):
        scores.setdefault(sid, 0.0)
    for sid, rank in ranks_a.items():
        scores[sid] += sem_w / (k_rrf + rank)
    for sid, rank in ranks_b.items():
        scores[sid] += bm25_w / (k_rrf + rank)
    fused = list(scores.items())
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
