"""Hybrid search dispatch — clip / bm25 / hybrid RRF over a single film.

Relocated from ``api/services/search.py`` during P1 / T10. The dispatcher
and its five private helpers form a self-contained block that orchestrates
three retrieval pipelines and folds them into one canonical DataFrame
shape (the same shape :func:`cinemateca.search.clip.search_text`
returns). Keeping the block in ``cinemateca.search`` means the HTTP /
service layer no longer owns retrieval orchestration — the route just
hands in a loaded :class:`SearchIndex` + (optional) :class:`BM25Index`
and asks for a mode.

Signature note: the function preserves the EXACT (positional ``index``,
keyword-only ``bm25 / retriever_mode / sem_w / bm25_w / rrf_k / ...``)
shape that lived in ``api.services.search`` so the existing route call
site (``api/routes/search.py``) and the 12 service-layer tests
(``tests/test_search_hybrid_service.py``) keep passing byte-identical.
A signature reshape into the deep-modules ``(query, film, mode)`` form
lands in T13 when the public ``cinemateca.search.find()`` verb is wired
up — verbatim move first, signature change behind a clean public verb
second.

Module hygiene: NO ``api.*`` imports. BM25 loading is the caller's job
(the legacy ``api/services/search._get_bm25_index_for_ctx`` resolves the
``cfg.search.bm25`` knobs and passes the pre-built index in). Keeps
this module loadable by tests and the planned ``cinemateca.search.find``
dispatcher without dragging in FastAPI app config.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from cinemateca.config import Settings
from cinemateca.retrieval.hybrid import DEFAULT_RRF_K, fuse_rrf, resolve_weights
from cinemateca.search.cache import SearchIndex
from cinemateca.search.clip import search_text

if TYPE_CHECKING:
    from cinemateca.retrieval.bm25 import BM25Index  # noqa: F401  — used in annotations

logger = logging.getLogger(__name__)

_VALID_RETRIEVERS = {"clip", "bm25", "hybrid"}


def resolve_retriever_args(
    cfg: Settings,
    retriever: str,
    sem_w: float | None,
    bm25_w: float | None,
) -> tuple[str, float, float, int]:
    """Normalise raw HTTP retriever args → ``(retriever, sem_w, bm25_w, rrf_k)``.

    Pulls defaults from ``cfg.search.hybrid_sem_w`` / ``hybrid_bm25_w`` and
    the optional ``cfg.search.bm25.rrf_k`` knob; clamps weights via
    :func:`resolve_weights`; falls back to ``"hybrid"`` on an unknown
    retriever string (matches the route's "don't 4xx on bookmarked URLs"
    policy — logged at WARNING). Either of ``sem_w`` / ``bm25_w`` being
    ``None`` uses the config default for that side.
    """
    if retriever not in _VALID_RETRIEVERS:
        logger.warning(
            "resolve_retriever_args: unknown retriever=%r — falling back to hybrid", retriever
        )
        retriever = "hybrid"
    defaults = (float(cfg.search.hybrid_sem_w), float(cfg.search.hybrid_bm25_w))
    sw, bw = resolve_weights(
        sem_w=sem_w if sem_w is not None else defaults[0],
        bm25_w=bm25_w if bm25_w is not None else defaults[1],
        defaults=defaults,
    )
    bm25_cfg = getattr(cfg.search, "bm25", None)
    rrf_k = int(getattr(bm25_cfg, "rrf_k", DEFAULT_RRF_K)) if bm25_cfg else DEFAULT_RRF_K
    return retriever, sw, bw, rrf_k


def search_hybrid(
    index: SearchIndex,
    *,
    bm25: BM25Index | None,
    query: str,
    tags: list[str],
    tag_index: dict,
    top_k: int,
    min_similarity: float,
    retriever_mode: str,
    sem_w: float,
    bm25_w: float,
    rrf_k: int = DEFAULT_RRF_K,
) -> pd.DataFrame:
    """Dispatch text search across one of three retrieval pipelines.

    Returns the same DataFrame shape ``search_text`` returns (columns:
    ``scene_id``, ``similarity``, plus the keyframe columns). The route
    layer cannot tell which mode produced the result — that's the
    whole point of the dispatcher.

    Args:
        index: per-film CLIP search index.
        bm25: per-film BM25 index, or ``None`` when ``retriever_mode`` is
            ``"clip"`` and BM25 wasn't loaded (cheap skip).
        retriever_mode: one of ``"clip" | "bm25" | "hybrid"``.
        sem_w, bm25_w: only consulted in ``"hybrid"`` mode.
        min_similarity: cosine floor for the CLIP path. Applied only on
            ``"clip"`` mode and on the CLIP side of ``"hybrid"`` (where
            it pre-filters before RRF). Not applied on ``"bm25"`` mode —
            BM25 scores are not in the cosine-similarity scale.

    Mode behaviour:
        * ``"clip"`` — delegates to :func:`search_text` unchanged.
          Regression pin for pre-M2 ordering.
        * ``"bm25"`` — runs BM25 only, formats the result the same way
          ``search_text`` would (mapping back to the index's metadata).
        * ``"hybrid"`` — runs both, fuses by weighted RRF, returns
          ``top_k`` by fused-score.

    Graceful fallback: when ``bm25`` is ``None`` or its underlying
    ``model`` is ``None`` (empty corpus), any non-``"clip"`` mode
    transparently degrades to CLIP-only — the UI degrades without
    raising, the route still serves a result.
    """
    if retriever_mode == "clip":
        return search_text(index, query, tags, tag_index, top_k, min_similarity)

    if bm25 is None or bm25.model is None:
        logger.info(
            "search_hybrid: bm25 index empty/None; falling back to clip (mode=%s requested)",
            retriever_mode,
        )
        return search_text(index, query, tags, tag_index, top_k, min_similarity)

    # Mirrors search_text's keyframe-density widening (4× ceiling).
    raw_k = top_k * 4

    # Compute the best-cosine row index per scene_id, IGNORING
    # min_similarity. Used by the BM25-only / fused backfill paths so a
    # BM25-only scene with multiple keyframes surfaces its best-cosine
    # keyframe rather than ``iloc[0]`` — parity with the CLIP-side
    # dedup (best-keyframe-per-scene). For pure-CLIP mode this is
    # unnecessary (search_text already dedupes); we compute it here
    # only for the bm25 / hybrid branches that follow.
    best_row_by_sid = _best_row_by_sid_from_embeddings(index, query)

    if retriever_mode == "bm25":
        # BM25 scores aren't cosine — min_similarity does not apply.
        hits = bm25.query(query, top_k=raw_k)
        return _bm25_hits_to_dataframe(
            hits, index, tags, tag_index, top_k, best_row_by_sid=best_row_by_sid
        )

    # retriever_mode == "hybrid". min_similarity flows through search_text
    # below, pre-filtering the CLIP side before RRF fusion.
    clip_df = search_text(index, query, tags, tag_index, raw_k, min_similarity)
    clip_ranked: list[tuple[int, float]] = (
        [(int(row.scene_id), float(row.similarity)) for row in clip_df.itertuples(index=False)]
        if not clip_df.empty
        else []
    )
    bm25_hits = bm25.query(query, top_k=raw_k)

    fused = fuse_rrf(clip_ranked, bm25_hits, sem_w=sem_w, bm25_w=bm25_w, k_rrf=rrf_k)[:top_k]

    return _fused_to_dataframe(
        fused, clip_df, index, tags, tag_index, top_k, best_row_by_sid=best_row_by_sid
    )


def _best_row_by_sid_from_embeddings(index: SearchIndex, query: str) -> dict[int, int]:
    """Map ``scene_id → kf_df row index of the highest-cosine keyframe``.

    The map is computed against the FULL embeddings matrix with NO
    min_similarity floor — its purpose is to pick the "best frame to
    display" for a scene that surfaces via BM25 alone (where the
    cosine never made it past the floor). Cosine ties are resolved by
    keeping the first row encountered (deterministic, matches the
    aggregate path's behaviour).

    Returns an empty dict on a degenerate index (no embedder / no
    rows). Callers treat the empty dict the same as "no best-row
    info" and fall back to ``iloc[0]`` per scene_id.
    """
    if (
        index.embeddings is None
        or index.kf_df is None
        or index.embedder is None
        or len(index.kf_df) == 0
    ):
        return {}
    text_vec = index.embedder.encode_text(query)  # type: ignore[union-attr]
    norm = float(np.linalg.norm(text_vec))
    text_vec = text_vec / (norm + 1e-12)
    scores = index.embeddings @ text_vec  # type: ignore[operator]
    best_score_by_sid: dict[int, float] = {}
    best_row_by_sid: dict[int, int] = {}
    for i, score in enumerate(scores):
        s = float(score)
        row = index.kf_df.iloc[i]  # type: ignore[union-attr]
        sid = int(row["scene_id"])
        prev = best_score_by_sid.get(sid)
        if prev is None or s > prev:
            best_score_by_sid[sid] = s
            best_row_by_sid[sid] = i
    return best_row_by_sid


def _allowed_scene_ids_for_tags(tags: list[str], tag_index: dict) -> set[int] | None:
    """AND-intersect the selected tags' scene-id memberships.

    Returns ``None`` when no tags are requested (no filter to apply).
    Returns a (possibly empty) ``set[int]`` of canonical scene_ids when
    tags ARE requested — including the empty case where the tag_index
    is empty or none of the tags exist in it. Callers must treat an
    empty set as "no scene matches" (NOT "no filter"), matching the
    AND-intersection contract used by ``SemanticSearch.combined`` and
    ``aggregate_search``'s CLIP path.

    The conversion is `int(sid)` rather than ``scene_id_key`` because
    BM25Index emits int sids and the per-film kf_df ``scene_id`` column
    is int — staying in the int domain avoids a redundant str cast for
    the per-film hybrid / bm25 paths. Mixed-type tag_index values
    (str manual + int LLM) round-trip safely through ``int()``.
    """
    if not tags:
        return None
    allowed: set[int] | None = None
    for t in tags:
        sids = tag_index.get(t, []) if isinstance(tag_index, dict) else []
        tag_sids: set[int] = set()
        for sid in sids:
            try:
                tag_sids.add(int(sid))
            except (TypeError, ValueError):
                continue
        allowed = tag_sids if allowed is None else (allowed & tag_sids)
    return allowed if allowed is not None else set()


def _bm25_hits_to_dataframe(
    hits: list[tuple[int, float]],
    index: SearchIndex,
    tags: list[str],
    tag_index: dict,
    top_k: int,
    *,
    best_row_by_sid: dict[int, int] | None = None,
) -> pd.DataFrame:
    """Materialise BM25-only hits into the ``search_text`` DataFrame shape.

    BM25 scores are not in CLIP's cosine-similarity scale, but the
    template only cares about ordering. We expose the BM25 score as
    ``similarity`` for shape-compat; routes that surface raw scores can
    distinguish via the ``retriever`` query param.

    Tag filter is AND-intersection (parity with the CLIP path and
    ``aggregate_search``). When ``tags`` are requested but no scene
    matches all of them — including the degenerate case where the film
    has no tag_index at all — the result is empty rather than
    silently-unfiltered.

    ``best_row_by_sid`` (optional) selects which keyframe row to surface
    for scenes with multiple keyframes. When provided, the highest-cosine
    row is used (parity with CLIP-side dedup); when absent the per-scene
    first row of ``kf_df`` is used. Callers in ``search_hybrid`` pass
    the map so a BM25-only multi-keyframe scene surfaces its best frame.
    """
    if not hits:
        return pd.DataFrame(columns=["scene_id", "similarity"])
    df = pd.DataFrame(hits, columns=["scene_id", "similarity"])
    allowed = _allowed_scene_ids_for_tags(tags, tag_index)
    if allowed is not None:
        df = df[df["scene_id"].isin(allowed)].reset_index(drop=True)
    if hasattr(index, "kf_df") and index.kf_df is not None and not df.empty:
        kf_picked = _pick_kf_rows_by_sid(index.kf_df, df["scene_id"].tolist(), best_row_by_sid)
        df = df.merge(kf_picked, on="scene_id", how="left")
    return df.head(top_k).reset_index(drop=True)


def _pick_kf_rows_by_sid(kf_df, sids: list[int], best_row_by_sid: dict[int, int] | None):
    """Pick one ``kf_df`` row per requested scene_id.

    Selection rule:
      * If ``best_row_by_sid`` is provided AND has an entry for the sid,
        return the row at that index (the highest-cosine keyframe).
      * Otherwise fall back to the first ``kf_df`` row matching the sid
        (deterministic — kf_df ordering is stable across loads).

    Returns a DataFrame with one row per input sid that exists in
    ``kf_df``. ``sids`` ordering is preserved.
    """
    rows = []
    for sid in sids:
        if best_row_by_sid and sid in best_row_by_sid:
            rows.append(kf_df.iloc[best_row_by_sid[sid]])
            continue
        mask = kf_df["scene_id"] == sid
        if mask.any():
            rows.append(kf_df[mask].iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def _fused_to_dataframe(
    fused: list[tuple[int, float]],
    clip_df: pd.DataFrame,
    index: SearchIndex,
    tags: list[str],
    tag_index: dict,
    top_k: int,
    *,
    best_row_by_sid: dict[int, int] | None = None,
) -> pd.DataFrame:
    """Materialise the fused ranking, reusing ``clip_df`` rows when present.

    BM25-only hits (scenes the CLIP top-K didn't surface) are back-filled
    from ``index.kf_df`` so every row carries the keyframe columns the
    template expects. The backfill triggers on ``filepath.isna()`` —
    the column name SemanticSearch.by_text / .by_image / .combined emit
    (NOT ``img_filename``, which never exists in the merged df and used
    to make this whole branch dead code).

    ``best_row_by_sid`` (optional) makes the backfill pick the
    highest-cosine keyframe of each BM25-only scene, matching how
    CLIP-side hits already dedup-by-best-keyframe. Without it, the
    backfill falls back to the first kf_df row per sid.

    Tag filter is AND-intersection (parity with CLIP / aggregate); when
    tags are requested but no scene matches all of them, the result is
    empty rather than silently-unfiltered.
    """
    if not fused:
        return pd.DataFrame(columns=["scene_id", "similarity"])
    fused_df = pd.DataFrame(fused, columns=["scene_id", "fused_score"])
    if not clip_df.empty:
        merged = fused_df.merge(
            clip_df.drop(columns=["similarity"], errors="ignore"),
            on="scene_id",
            how="left",
        )
    else:
        merged = fused_df
    merged["similarity"] = merged["fused_score"]
    merged = merged.drop(columns=["fused_score"])
    allowed = _allowed_scene_ids_for_tags(tags, tag_index)
    if allowed is not None:
        merged = merged[merged["scene_id"].isin(allowed)].reset_index(drop=True)
    if hasattr(index, "kf_df") and index.kf_df is not None and not merged.empty:
        # Backfill keyframe metadata for BM25-only hits. ``filepath`` is
        # the column SemanticSearch emits; when clip_df contributed
        # nothing for a scene_id, the left-merge above leaves it NaN
        # and we patch from index.kf_df. If ``filepath`` itself is
        # missing from the merged frame (clip_df was empty), every row
        # needs patching.
        if "filepath" in merged.columns:
            missing_mask = merged["filepath"].isna()
        else:
            missing_mask = pd.Series([True] * len(merged), index=merged.index)
        missing = merged[missing_mask]
        if not missing.empty:
            kf_picked = _pick_kf_rows_by_sid(
                index.kf_df, missing["scene_id"].tolist(), best_row_by_sid
            )
            patched = missing[["scene_id", "similarity"]].merge(
                kf_picked, on="scene_id", how="left"
            )
            merged = pd.concat(
                [merged[~merged["scene_id"].isin(patched["scene_id"])], patched],
                ignore_index=True,
            )
            merged = merged.sort_values("similarity", ascending=False).reset_index(drop=True)
    return merged.head(top_k).reset_index(drop=True)
