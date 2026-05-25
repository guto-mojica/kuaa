"""Cross-film aggregate search — global retrieval lists over the library.

Relocated from ``api/services/search.py`` during P1 / T11. The function
walks every registered film, builds per-film CLIP + (optional) BM25
ranked lists, concatenates them into two GLOBAL ranked lists keyed by
``(film_slug, scene_id)``, then dispatches by retriever mode:

  * ``"clip"``   — global CLIP list, sorted by cosine.
  * ``"bm25"``   — global BM25 list, sorted by raw BM25 score.
  * ``"hybrid"`` — weighted RRF fusion of the two global lists.

Signature note: the function preserves the EXACT ``cfg``-taking, keyword-
only signature that lived in ``api.services.search`` so the existing
route call site (``api/routes/search.py``) and the 18 cross-film tests
(``test_multi_film_search.py``, ``test_aggregate_search_hybrid.py``,
``test_p1_search_snapshot.py``) keep passing byte-identical. A public
typed ``aggregate(query, *, cfg, ...)`` wrapper lands in T13 — verbatim
move first, signature reshape behind a stable public surface second.

Module hygiene note: this module makes a LAZY call into
``api.services.search`` for two helpers — ``_get_embedder`` (the CLIP
text-encoder factory) and ``_get_search_index`` (the per-film
``SearchIndex`` loader). The lazy attribute access (NOT a top-level
``from ... import``) is intentional: tests across the suite monkeypatch
``api.services.search._get_embedder`` / ``_get_search_index`` to avoid
loading the real ~4 s CLIP model, and we MUST keep those monkeypatches
on the call path. Reading the attribute from the module at call time
(rather than binding the function reference at import time) preserves
that contract byte-for-byte. The import is also lazy because
``api/services/search.py`` re-exports this module's :func:`aggregate_search`
back to its old name — a top-level ``from api.services import search``
here would deadlock the module-load order. T14 collapses both shims
into the public ``cinemateca.search.find()`` verb and the lazy
attribute reads disappear.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from api.services.catalog import derive_fps, load_json, load_tag_index, to_smpte
from api.services.film_context import FilmContext
from cinemateca.retrieval.hybrid import DEFAULT_RRF_K, fuse_rrf
from cinemateca.scene_ids import normalize_tag_index, scene_id_key
from cinemateca.search.cache import IndexStatus
from cinemateca.search.types import (
    Filters,
    Hit,
    HybridWeights,
    Query,
    SearchMode,
    SearchResult,
)

logger = logging.getLogger(__name__)


def aggregate_search(
    cfg: Any,
    *,
    query: str,
    modality: str,
    top_k: int,
    tags: list[str] | None = None,
    min_similarity: float = 0.0,
    retriever_mode: str = "clip",
    sem_w: float = 0.70,
    bm25_w: float = 0.30,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[dict]:
    """Run cross-film search using GLOBAL retrieval lists, not per-film.

    For ``modality == "text"`` only in this plan; image / audio / fusion
    modalities are wired in Plans 3-5.

    Pipeline (all three modes):
      1. Walk every registered film. For each, compute the per-film
         CLIP cosine list (best keyframe per scene_id, descending) and,
         for non-``clip`` modes, the per-film BM25 hit list. Skip films
         whose CLIP index is missing/corrupt or whose tag-intersection
         (when ``tags`` is non-empty) is empty.
      2. Concatenate every film's lists into two GLOBAL ranked lists
         keyed by ``(film_slug, scene_id)`` — the global CLIP list
         sorted by cosine, the global BM25 list sorted by raw BM25 score.
      3. Dispatch by ``retriever_mode``:

         * ``"clip"`` — surface the global CLIP list (cosine is cross-
           film-comparable; the legacy path's score and ordering are
           preserved byte-for-byte, since this is the same set of items
           in the same order).
         * ``"bm25"`` — surface the global BM25 list. Per-film IDF
           variance means raw BM25 scores are only approximately
           comparable cross-film, but rank-sort by raw score is a
           defensible heuristic (and identical to the legacy code).
         * ``"hybrid"`` — fuse the global CLIP and global BM25 lists
           via weighted RRF. The previous implementation ran per-film
           RRF and then sorted across films by raw RRF score, which is
           DEGENERATE: every film's per-film rank-1 contribution gets
           the same score ``1/(rrf_k+1)``, so the cross-film top-K
           tied scores and ordering was decided by film-iteration
           order, not signal strength. Global RRF assigns each item
           a single global rank per side, breaking the tie.

      4. Materialise the top_k as hit dicts. Keyframe filepath uses the
         per-film best-cosine row when available (the same
         ``best_row_by_sid`` map the CLIP pass built); pure BM25-only
         scenes fall back to the first kf_df row for that scene_id —
         deterministic.

    ``tags`` AND-intersects across selected tags via the same
    ``normalize_tag_index`` / ``scene_id_key`` pipeline used by
    ``SemanticSearch.combined`` — applied identically to the CLIP and
    BM25 sides so ``?retriever=bm25&tags=outdoor`` cannot silently
    ignore the tag.

    ``min_similarity`` floors the CLIP cosine before it enters the
    global list. It is NOT applied to BM25 scores (different scale) or
    to fused RRF scores (different scale again) — same contract as the
    legacy code.
    """
    # Lazy attribute reads on api.services.search so the existing
    # monkeypatches (test_multi_film_search / test_aggregate_search_hybrid /
    # test_p1_search_snapshot all setattr on the legacy module path) keep
    # hitting the call sites below. ``_get_search_index`` and
    # ``_get_embedder`` migrate under cinemateca.search in a follow-up;
    # until then this is the documented contract.
    from api.services import search as _legacy_search
    from cinemateca.library import scan_library

    if modality != "text":
        raise NotImplementedError(
            f"modality={modality!r} lands in a later plan; only 'text' is supported here"
        )

    library_dir = Path(cfg.paths.library_dir)
    # Materialise the film list BEFORE loading the embedder.  When the library
    # is empty (0 registered films) we return early and avoid the ~4 s CLIP
    # model initialisation that _get_embedder triggers.
    films = list(scan_library(library_dir))
    if not films:
        return []

    embedder = _legacy_search._get_embedder(cfg)

    text_vec: np.ndarray = embedder.encode_text(query)  # type: ignore[union-attr]
    norm = float(np.linalg.norm(text_vec))
    text_vec = text_vec / (norm + 1e-12)

    selected_tags = list(tags) if tags else []
    # Widen the per-film retrieval window so the global pool stays dense
    # enough to fill ``top_k`` after fusion. ``raw_k`` per film is
    # sufficient because the global lists union all films' top-raw_k —
    # the global rank-1 from any film is guaranteed to enter.
    raw_k = max(top_k * 4, 1)

    logger.info(
        "aggregate_search: query=%r films=%d top_k=%d tags=%s min_sim=%.3f "
        "retriever=%s sem_w=%.2f bm25_w=%.2f rrf_k=%d",
        query,
        len(films),
        top_k,
        selected_tags or None,
        min_similarity,
        retriever_mode,
        sem_w,
        bm25_w,
        rrf_k,
    )

    # Per-film state cache. Keys are film slugs; values carry every
    # object the materialisation step (Phase 4) needs to build a hit
    # dict — film metadata, the CLIP index (kf_df), best-row-by-sid
    # map, fps, and the scene-id → kf_meta lookup. We populate it once
    # per film during Phase 1 so Phase 4 is pure look-up.
    PerFilm = dict  # alias for readability — kept structural to avoid a dataclass for one local
    per_film: dict[str, PerFilm] = {}
    # GLOBAL ranked lists. Keys are (film_slug, scene_id) tuples.
    # ``fuse_rrf`` only requires hashable keys; the int-only type hint
    # is a documentation choice, not a runtime constraint.
    global_clip: list[tuple[tuple[str, int], float]] = []
    global_bm25: list[tuple[tuple[str, int], float]] = []

    for film in films:
        try:
            idx = _legacy_search._get_search_index(cfg, film.slug)
        except ValueError as exc:
            logger.warning("aggregate_search: skip film %s — %s", film.slug, exc)
            continue
        if idx.status is not IndexStatus.OK:
            logger.info(
                "aggregate_search: skip film %s — index status %s",
                film.slug,
                idx.status,
            )
            continue
        ctx = FilmContext.for_film(cfg, film.slug)
        kf_meta = load_json(ctx.metadata_dir / "keyframes_metadata.json") or []
        fps = derive_fps(kf_meta)
        meta_by_scene = {e["scene_id"]: e for e in kf_meta if "scene_id" in e}

        # Tag pre-filter (AND intersection across selected tags). Mirrors
        # SemanticSearch.combined: normalize the raw tag_index to canonical
        # str scene ids, intersect membership sets, skip the film entirely
        # if any selected tag is missing or the intersection is empty.
        allowed_scene_keys: set[str] | None = None
        if selected_tags:
            raw_index = load_tag_index(ctx.metadata_dir)
            norm_index = normalize_tag_index(raw_index)
            allowed_scene_keys = set(norm_index.get(selected_tags[0], set()))
            for t in selected_tags[1:]:
                allowed_scene_keys &= set(norm_index.get(t, set()))
            if not allowed_scene_keys:
                continue

        scores: np.ndarray = idx.embeddings @ text_vec  # type: ignore[operator]

        # CLIP-side ranked list — `(scene_id, cosine_score)` descending.
        # Best-keyframe-per-scene: a single scene may have multiple
        # keyframes (Phase-1 density), so the same scene_id can appear N
        # times in ``scores`` at different rows. Keep the row index with
        # the HIGHEST cosine per scene_id so the surfaced
        # ``keyframe_path`` points at the actual best-matching frame.
        best_score_by_sid: dict[int, float] = {}
        best_row_by_sid: dict[int, int] = {}
        for i, score in enumerate(scores):
            s = float(score)
            if s < min_similarity:
                continue
            row = idx.kf_df.iloc[i]  # type: ignore[union-attr]
            sid = int(row["scene_id"])
            if allowed_scene_keys is not None and scene_id_key(sid) not in allowed_scene_keys:
                continue
            prev = best_score_by_sid.get(sid)
            if prev is None or s > prev:
                best_score_by_sid[sid] = s
                best_row_by_sid[sid] = i
        clip_ranked: list[tuple[int, float]] = sorted(
            best_score_by_sid.items(), key=lambda p: p[1], reverse=True
        )[:raw_k]

        # BM25-side ranked list. ``"clip"`` mode skips BM25 loading
        # entirely (no need to pay the disk read for a corpus we'll
        # ignore). A film whose BM25 corpus is empty contributes
        # nothing to the global BM25 list — in hybrid mode that scene
        # still surfaces via CLIP-only contribution, in pure-bm25 mode
        # it surfaces nothing (which is correct: BM25 has no signal).
        bm25_hits: list[tuple[int, float]] = []
        if retriever_mode != "clip":
            try:
                bm25 = _legacy_search._get_bm25_index_for_ctx(ctx)
            except (FileNotFoundError, OSError, ValueError):
                # Narrow set of loader failure modes — anything else is
                # a programming bug we want to surface, not silently
                # absorb. The film simply contributes no BM25 entries.
                logger.warning(
                    "aggregate_search: bm25 loader failed for %s; "
                    "contributing no BM25 entries for this film",
                    film.slug,
                    exc_info=True,
                )
                bm25 = None
            if bm25 is None or bm25.model is None:
                logger.info(
                    "aggregate_search: film=%s bm25 empty; no BM25 entries " "(mode=%s requested)",
                    film.slug,
                    retriever_mode,
                )
            else:
                bm25_hits = bm25.query(query, top_k=raw_k)
                if allowed_scene_keys is not None:
                    bm25_hits = [
                        (sid, s) for sid, s in bm25_hits if scene_id_key(sid) in allowed_scene_keys
                    ]

        per_film[film.slug] = {
            "film": film,
            "kf_df": idx.kf_df,
            "best_row_by_sid": best_row_by_sid,
            "fps": fps,
            "meta_by_scene": meta_by_scene,
        }
        for sid, s in clip_ranked:
            global_clip.append(((film.slug, sid), s))
        for sid, s in bm25_hits:
            global_bm25.append(((film.slug, sid), s))

        if scores.size:
            top3 = np.sort(scores)[-3:][::-1]
            logger.info(
                "aggregate_search: film=%s n_vectors=%d top3=%s "
                "clip_n=%d bm25_n=%d (retriever=%s)",
                film.slug,
                int(scores.size),
                [round(float(s), 3) for s in top3],
                len(clip_ranked),
                len(bm25_hits),
                retriever_mode,
            )

    # Phase 2: build globally-ranked lists.
    global_clip.sort(key=lambda p: p[1], reverse=True)
    global_bm25.sort(key=lambda p: p[1], reverse=True)

    # Phase 3: dispatch by mode. ``ranked`` is the unified output:
    # a list of ``((film_slug, scene_id), score)`` pairs, top first.
    ranked: list[tuple[tuple[str, int], float]]
    if retriever_mode == "clip":
        ranked = global_clip
    elif retriever_mode == "bm25":
        ranked = global_bm25
    else:  # "hybrid"
        ranked = fuse_rrf(
            global_clip,
            global_bm25,
            sem_w=sem_w,
            bm25_w=bm25_w,
            k_rrf=rrf_k,
        )

    # Phase 4: materialise hit dicts. Keys are already unique
    # ((film_slug, scene_id)) so no dedupe pass is needed.
    all_hits: list[dict] = []
    for (slug, sid), score in ranked[:top_k]:
        state = per_film.get(slug)
        if state is None:  # defensive — every key came from per_film
            continue
        kf_df = state["kf_df"]
        best_i = state["best_row_by_sid"].get(sid)
        if best_i is not None:
            row = kf_df.iloc[best_i]
        else:
            # BM25-only scene whose cosine was below ``min_similarity``
            # or is otherwise absent from the CLIP-side map. Fall back
            # to the first kf_df row for that scene_id — deterministic
            # because kf_df row order is stable across loads.
            row_mask = kf_df["scene_id"] == sid
            if not row_mask.any():
                continue
            row = kf_df[row_mask].iloc[0]
        meta = state["meta_by_scene"].get(sid)
        start_s = float(meta.get("start_time_s") or 0.0) if meta else 0.0
        timecode = to_smpte(start_s, state["fps"]) if start_s > 0 else ""
        all_hits.append(
            {
                "film_slug": slug,
                "film_title": state["film"].title,
                "scene_id": sid,
                "score": float(score),
                "keyframe_path": str(row["filepath"]),
                "timecode": timecode,
            }
        )

    logger.info(
        "aggregate_search: query=%r global_clip=%d global_bm25=%d " "returned=%d top_score=%.6f",
        query,
        len(global_clip),
        len(global_bm25),
        len(all_hits),
        float(all_hits[0]["score"]) if all_hits else 0.0,
    )
    return all_hits


# ── Typed public wrapper (T13) ────────────────────────────────────────────────
# ``aggregate(query, *, cfg, mode, top_k, filters, weights)`` is the public
# verb. It wraps the legacy ``aggregate_search`` (above, dict-returning)
# with the locked P1 API surface — typed in, typed out. P2 replaces the
# ``cfg=`` argument with ``library=Library`` once the Library type
# lands; the rest of the signature stays.
#
# ``no_index`` resolution uses :func:`api.services.search.has_indexed_films`
# directly. The carve-out
# ``cinemateca.search.aggregate -> api.services.search`` already exists
# in ``.importlinter`` from T11 (the lazy ``_get_embedder`` /
# ``_get_search_index`` reads) — no additional rule needed for T13.


def aggregate(
    query: Query,
    *,
    cfg: Any,
    mode: SearchMode = "clip",
    top_k: int = 20,
    filters: Filters | None = None,
    weights: HybridWeights | None = None,
) -> SearchResult:
    """Public verb: cross-film aggregate search.

    Wraps the legacy :func:`aggregate_search` (dict-returning) with the
    typed API. P1 supports text queries only — image / audio / fusion
    modalities land in later plans. ``cfg`` is the existing app-config
    handle; P2 replaces it with ``library=Library``.

    Returns a typed :class:`SearchResult`. ``no_index=True`` carries the
    empty-library / unindexed-films signal so the caller renders the
    no-index UI state.
    """
    if query.text is None:
        raise NotImplementedError(
            "aggregate() supports text queries only in P1; "
            "image / audio / fusion modalities land in later plans."
        )
    filters = filters or Filters()
    weights = weights or HybridWeights()
    raw = aggregate_search(
        cfg,
        query=query.text,
        modality="text",
        top_k=top_k,
        tags=list(filters.tags) or None,
        min_similarity=filters.min_similarity,
        retriever_mode=mode,
        sem_w=weights.sem_w,
        bm25_w=weights.bm25_w,
        rrf_k=weights.rrf_k,
    )
    hits = [
        Hit(
            scene_id=int(h["scene_id"]),
            score=float(h["score"]),
            keyframe_path=str(h.get("keyframe_path", "")),
            film_slug=h.get("film_slug"),
            film_title=h.get("film_title"),
            timecode=h.get("timecode", ""),
        )
        for h in raw
    ]
    if hits:
        no_index = False
    else:
        from api.services.search import has_indexed_films

        no_index = not has_indexed_films(cfg)
    return SearchResult(
        hits=hits,
        mode=mode,
        weights=weights if mode == "hybrid" else None,
        query=query,
        no_index=no_index,
    )


def aggregate_hits_to_template_dicts(cfg: Any, hits: list[dict]) -> list[dict]:
    """Convert ``aggregate_search`` raw hits to ``.b-card``-shaped template dicts.

    Relocated from the aggregate path of ``api/routes/search.py`` (T15).
    The ``data_dir`` MUST be the media-mount root (``cfg.paths.data_dir``),
    not ``library_dir`` — otherwise ``keyframe_url``'s ``relative_to``
    check fails for filepaths stored under ``data/frames/...`` or
    ``data/library/<slug>/...`` and the template gets
    ``img_url=None`` for every row.

    Aggregate hits already carry ``film_slug`` / ``film_title`` /
    ``timecode`` (computed by ``aggregate_search``). Template uses
    ``r.similarity``; aggregate_search emits ``score`` (cosine over the
    per-film index) — aliased here so the same ``partials/search_results.html``
    works for the per-film and aggregate paths.
    """
    from api.services.catalog import keyframe_url

    data_dir = Path(cfg.paths.data_dir).resolve()
    return [
        {
            "film_slug": h["film_slug"],
            "film_title": h["film_title"],
            "scene_id": h["scene_id"],
            "similarity": h["score"],
            "img_url": keyframe_url(h["keyframe_path"], data_dir),
            "timecode": h["timecode"],
        }
        for h in hits
    ]
