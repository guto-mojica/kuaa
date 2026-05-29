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

T13 (P3.D.1): ``_get_embedder``, ``_get_search_index``, and
``has_indexed_films`` now live here. ``api.services.search`` re-exports
them for backward compatibility. Tests should monkeypatch
``cinemateca.search.aggregate._get_embedder`` / ``_get_search_index``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

from cinemateca.config import Settings

from cinemateca.library import (
    FilmContext,
    derive_fps,
    load_json,
    load_tag_index,
    to_smpte,
)
from cinemateca.retrieval.hybrid import DEFAULT_RRF_K
from cinemateca.retrieval.tokenize import tokenize
from cinemateca.scene_ids import normalize_tag_index, scene_id_key
from cinemateca.search.cache import IndexStatus, SearchIndex, load_index
from cinemateca.search.types import (
    Filters,
    Hit,
    HybridWeights,
    Query,
    SearchMode,
    SearchResult,
)

logger = logging.getLogger(__name__)

# Canonical filenames for the per-film CLIP index. Mirror
# ``config/default.yaml`` → ``embeddings.*``; used as defaults when
# ``cfg.embeddings`` is absent (unit tests with minimal configs).
_DEFAULT_EMBEDDINGS_FILENAME = "keyframe_embeddings.npy"
_DEFAULT_MAPPING_FILENAME = "index_mapping.json"
_SCENE_ID_FROM_PATH_RE = re.compile(r"Scene-(\d+)", flags=re.IGNORECASE)


def _get_embedder(cfg: Settings) -> Any:
    """Return a fresh image embedder via the registry. Module-scope so unit
    tests monkeypatch ``cinemateca.search.aggregate._get_embedder`` to avoid
    loading the real model.

    Dispatch honours ``cfg.models.image_embedder`` so the SigLIP-multilingual
    backend (M3 pre-flight Task 4.2 flip) returns a SigLIP encoder whose
    output dim matches the on-disk index, instead of the previous hardcoded
    ``OpenClipEmbedder()`` which 500'd on dim mismatch against a 1024-dim
    SigLIP index.
    """
    from cinemateca.models.registry import get_image_embedder

    return get_image_embedder(cfg)


def _get_search_index(cfg: Settings, slug: str) -> SearchIndex:
    """Return the (cached) :class:`SearchIndex` for ``slug``. Reads
    ``cfg.embeddings.*`` filenames when present, otherwise falls back
    to the module-level defaults for minimal test configs.
    """
    emb_cfg = getattr(cfg, "embeddings", None)
    embeddings_filename = (
        getattr(emb_cfg, "filename", _DEFAULT_EMBEDDINGS_FILENAME)
        if emb_cfg is not None
        else _DEFAULT_EMBEDDINGS_FILENAME
    )
    mapping_filename = (
        getattr(emb_cfg, "mapping_filename", _DEFAULT_MAPPING_FILENAME)
        if emb_cfg is not None
        else _DEFAULT_MAPPING_FILENAME
    )
    ctx = FilmContext.for_film(cfg, slug)
    return load_index(
        ctx, embeddings_filename=embeddings_filename, mapping_filename=mapping_filename, cfg=cfg
    )


def _get_bm25_index_for_ctx_with_cfg(cfg: Settings, ctx: FilmContext) -> Any:
    """Load + cache the BM25 index for one film.

    Resolves ``cfg.search.bm25`` tunables (``stopwords_lang`` / ``k1`` /
    ``b``) directly from the supplied ``cfg`` object — no ``api.deps``
    dependency needed because aggregate callers already hold ``cfg``.
    """
    from cinemateca.search.bm25 import bm25_index_for_ctx

    bm25_cfg = getattr(cfg.search, "bm25", None) if hasattr(cfg, "search") else None
    stopwords_lang = getattr(bm25_cfg, "stopwords_lang", None) if bm25_cfg else None
    k1 = float(getattr(bm25_cfg, "k1", 1.5)) if bm25_cfg else 1.5
    b = float(getattr(bm25_cfg, "b", 0.75)) if bm25_cfg else 0.75
    include_transcripts = bool(getattr(bm25_cfg, "include_transcripts", True)) if bm25_cfg else True
    return bm25_index_for_ctx(
        ctx,
        stopwords_lang=stopwords_lang,
        k1=k1,
        b=b,
        include_transcripts=include_transcripts,
    )


def _scene_id_from_visual_record(record: dict[str, Any]) -> int | None:
    """Best-effort scene id extraction for visual-analysis rows."""
    sid = record.get("scene_id")
    if sid is not None:
        try:
            return int(sid)
        except (TypeError, ValueError):
            return None
    frame_path = str(record.get("frame_path") or record.get("filepath") or "")
    match = _SCENE_ID_FROM_PATH_RE.search(frame_path)
    if match:
        return int(match.group(1))
    return None


def _tokens_for_value(value: Any) -> list[str]:
    """Tokenize nested metadata values without assuming a fixed schema."""
    if value is None:
        return []
    if isinstance(value, str):
        return tokenize(value)
    if isinstance(value, (int, float, bool)):
        return tokenize(str(value))
    if isinstance(value, (list, tuple, set)):
        item_tokens: list[str] = []
        for item in value:
            item_tokens.extend(_tokens_for_value(item))
        return item_tokens
    if isinstance(value, dict):
        dict_tokens: list[str] = []
        for item in value.values():
            dict_tokens.extend(_tokens_for_value(item))
        return dict_tokens
    return tokenize(str(value))


def _phrase_match_score(
    value: Any, query_tokens: list[str], *, exact: float, contains: float
) -> float:
    tokens = _tokens_for_value(value)
    if not tokens or not query_tokens:
        return 0.0
    q = tuple(query_tokens)
    if tuple(tokens) == q:
        return exact
    token_set = set(tokens)
    if all(t in token_set for t in query_tokens):
        return contains
    return 0.0


def _metadata_scores_for_query(
    *,
    query: str,
    descriptions: list[dict[str, Any]],
    tag_index: dict[str, Any],
    visual_rows: list[dict[str, Any]],
) -> dict[int, float]:
    """Return exact metadata/object match scores keyed by scene id.

    This is intentionally lexical. SigLIP handles broad semantic similarity;
    this signal protects short object queries such as ``dog`` where exact tags,
    visual object classes, and description/object fields are stronger evidence
    than a weak visual cosine rank.
    """
    query_tokens = tokenize(query)
    if not query_tokens or len(query_tokens) > 4:
        return {}

    scores: dict[int, float] = {}

    def add(sid: Any, delta: float) -> None:
        if delta <= 0:
            return
        try:
            sid_int = int(sid)
        except (TypeError, ValueError):
            return
        scores[sid_int] = scores.get(sid_int, 0.0) + delta

    for tag, sids in tag_index.items():
        tag_score = _phrase_match_score(tag, query_tokens, exact=0.25, contains=0.1)
        if tag_score <= 0 or not isinstance(sids, (list, tuple, set)):
            continue
        for sid in sids:
            add(sid, tag_score)

    for entry in descriptions:
        sid = entry.get("scene_id")
        if sid is None:
            continue
        desc_score = _phrase_match_score(
            entry.get("description"), query_tokens, exact=12.0, contains=12.0
        )
        action_score = _phrase_match_score(
            entry.get("people_action"), query_tokens, exact=2.0, contains=2.0
        )
        add(sid, desc_score)
        add(sid, action_score)
        has_description_evidence = desc_score > 0.0

        # Structured generated labels support the written description/action.
        # Alone, they are intentionally weak: these labels can contain loose
        # object guesses that are noisier than the prose or detector output.
        structured_exact = 3.0 if has_description_evidence else 1.0
        structured_contains = 2.0 if has_description_evidence else 0.5
        for key in ("objects", "tags"):
            add(
                sid,
                _phrase_match_score(
                    entry.get(key),
                    query_tokens,
                    exact=structured_exact,
                    contains=structured_contains,
                ),
            )
        for key in ("setting", "location"):
            add(sid, _phrase_match_score(entry.get(key), query_tokens, exact=2.0, contains=1.0))
        raw = entry.get("_raw_responses")
        if isinstance(raw, dict):
            add(
                sid,
                _phrase_match_score(
                    raw.get("objects"),
                    query_tokens,
                    exact=structured_exact,
                    contains=structured_contains,
                ),
            )

    for row in visual_rows:
        sid = _scene_id_from_visual_record(row)
        if sid is None:
            continue
        obj = row.get("object_detection")
        if not isinstance(obj, dict):
            continue
        for detected in obj.get("objects") or []:
            if isinstance(detected, dict):
                add(
                    sid,
                    _phrase_match_score(
                        detected.get("class"), query_tokens, exact=10.0, contains=7.0
                    ),
                )
        class_counts = obj.get("class_counts")
        if isinstance(class_counts, dict):
            for cls, count in class_counts.items():
                try:
                    n = max(1.0, float(count))
                except (TypeError, ValueError):
                    n = 1.0
                add(
                    sid,
                    min(12.0, n * _phrase_match_score(cls, query_tokens, exact=10.0, contains=7.0)),
                )

    return scores


from cinemateca.search._aggregate.fusion import fuse_global_rrf as _fuse_rrf_many


def has_indexed_films(cfg: Settings) -> bool:
    """``True`` iff at least one registered film has an OK :class:`SearchIndex`.

    Lets the route distinguish "no indexed films yet" (run the pipeline)
    from "indexed films exist but the query matched nothing" (no results).
    """
    from cinemateca.library import scan_library

    library_dir = Path(cfg.paths.library_dir)
    for film in scan_library(library_dir):
        try:
            idx = _get_search_index(cfg, film.slug)
        except ValueError:
            continue
        if idx.status is IndexStatus.OK:
            return True
    return False


def aggregate_search(
    cfg: Settings,
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

    # Pre-scan: skip embedder init when no film has a valid index.
    # Uses _get_search_index (which is monkeypatched in tests and cached
    # in production) rather than a bare disk check so test fixtures that
    # stub the index are respected.
    valid_slugs: set[str] = set()
    for _f in films:
        try:
            _idx = _get_search_index(cfg, _f.slug)
        except ValueError:
            continue
        if _idx.status is IndexStatus.OK:
            valid_slugs.add(_f.slug)

    if not valid_slugs:
        return []

    embedder = _get_embedder(cfg)

    text_vec: np.ndarray = embedder.encode_text(query)
    norm = float(np.linalg.norm(text_vec))
    text_vec = text_vec / (norm + 1e-12)

    selected_tags = list(tags) if tags else []
    # Widen the per-film retrieval window so the global pool stays dense
    # enough to fill ``top_k`` after fusion. Short object queries often need
    # lexical/object matches rescued from below the weak SigLIP top ranks.
    raw_k = max(top_k * 12, 50, 1)

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
    per_film: dict[str, dict[str, Any]] = {}
    # GLOBAL ranked lists. Keys are (film_slug, scene_id) tuples.
    # ``fuse_rrf`` only requires hashable keys; the int-only type hint
    # is a documentation choice, not a runtime constraint.
    global_clip: list[tuple[tuple[str, int], float]] = []
    global_bm25: list[tuple[tuple[str, int], float]] = []
    global_metadata: list[tuple[tuple[str, int], float]] = []

    for film in films:
        try:
            idx = _get_search_index(cfg, film.slug)
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
        kf_meta_data = load_json(ctx.metadata_dir / "keyframes_metadata.json") or []
        kf_meta = kf_meta_data if isinstance(kf_meta_data, list) else []
        fps = derive_fps(kf_meta)
        meta_by_scene = {e["scene_id"]: e for e in kf_meta if "scene_id" in e}
        descriptions_data = load_json(ctx.metadata_dir / "scene_descriptions.json") or []
        descriptions = descriptions_data if isinstance(descriptions_data, list) else []
        tag_index = load_tag_index(ctx.metadata_dir) or {}
        visual_data = load_json(ctx.metadata_dir / "visual_analysis.json") or []
        visual_rows = visual_data if isinstance(visual_data, list) else []
        metadata_scores = (
            _metadata_scores_for_query(
                query=query,
                descriptions=descriptions,
                tag_index=tag_index,
                visual_rows=visual_rows,
            )
            if retriever_mode == "hybrid"
            else {}
        )
        metadata_ranked: list[tuple[int, float]] = sorted(
            metadata_scores.items(), key=lambda p: p[1], reverse=True
        )[:raw_k]

        # Tag pre-filter (AND intersection across selected tags). Mirrors
        # SemanticSearch.combined: normalize the raw tag_index to canonical
        # str scene ids, intersect membership sets, skip the film entirely
        # if any selected tag is missing or the intersection is empty.
        allowed_scene_keys: set[str] | None = None
        if selected_tags:
            norm_index = normalize_tag_index(tag_index)
            allowed_scene_keys = set(norm_index.get(selected_tags[0], set()))
            for t in selected_tags[1:]:
                allowed_scene_keys &= set(norm_index.get(t, set()))
            if not allowed_scene_keys:
                continue

        embeddings: Any = idx.embeddings
        kf_df: Any = idx.kf_df
        scores: np.ndarray = embeddings @ text_vec

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
            row = kf_df.iloc[i]
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
                bm25 = _get_bm25_index_for_ctx_with_cfg(cfg, ctx)
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
            "kf_df": kf_df,
            "best_row_by_sid": best_row_by_sid,
            "fps": fps,
            "meta_by_scene": meta_by_scene,
        }
        for sid, s in clip_ranked:
            global_clip.append(((film.slug, sid), s))
        for sid, s in bm25_hits:
            global_bm25.append(((film.slug, sid), s))
        for sid, s in metadata_ranked:
            if allowed_scene_keys is None or scene_id_key(sid) in allowed_scene_keys:
                global_metadata.append(((film.slug, sid), s))

        if scores.size:
            top3 = np.sort(scores)[-3:][::-1]
            logger.info(
                "aggregate_search: film=%s n_vectors=%d top3=%s "
                "clip_n=%d bm25_n=%d metadata_n=%d (retriever=%s)",
                film.slug,
                int(scores.size),
                [round(float(s), 3) for s in top3],
                len(clip_ranked),
                len(bm25_hits),
                len(metadata_ranked),
                retriever_mode,
            )

    # Phase 2: build globally-ranked lists.
    global_clip.sort(key=lambda p: p[1], reverse=True)
    global_bm25.sort(key=lambda p: p[1], reverse=True)
    global_metadata.sort(key=lambda p: p[1], reverse=True)

    # Phase 3: dispatch by mode. ``ranked`` is the unified output:
    # a list of ``((film_slug, scene_id), score)`` pairs, top first.
    ranked: list[tuple[tuple[str, int], float]]
    if retriever_mode == "clip":
        ranked = global_clip
    elif retriever_mode == "bm25":
        ranked = global_bm25
    else:  # "hybrid"
        if global_metadata:
            metadata_w = 0.65
            residual_w = 1.0 - metadata_w
            retrieval_total = max(float(sem_w) + float(bm25_w), 1e-12)
            ranked = _fuse_rrf_many(
                [
                    (global_metadata, metadata_w),
                    (global_clip, residual_w * float(sem_w) / retrieval_total),
                    (global_bm25, residual_w * float(bm25_w) / retrieval_total),
                ],
                k_rrf=rrf_k,
            )
        else:
            ranked = _fuse_rrf_many(
                [
                    (global_clip, float(sem_w)),
                    (global_bm25, float(bm25_w)),
                ],
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
        "aggregate_search: query=%r global_clip=%d global_bm25=%d global_metadata=%d "
        "returned=%d top_score=%.6f",
        query,
        len(global_clip),
        len(global_bm25),
        len(global_metadata),
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


def aggregate(
    query: Query,
    *,
    cfg: Settings,
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
        no_index = not has_indexed_films(cfg)
    return SearchResult(
        hits=hits,
        mode=mode,
        weights=weights if mode == "hybrid" else None,
        query=query,
        no_index=no_index,
    )


def aggregate_hits_to_template_dicts(cfg: Settings, hits: list[dict]) -> list[dict]:
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
    from cinemateca.library import keyframe_url

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
