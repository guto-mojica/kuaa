"""Search service — CLIP semantic-search domain logic + index validation.

This module owns what used to live inline in ``api/routes/search.py``:

  * loading the on-disk CLIP index (``.npy`` embeddings + the
    ``index_mapping.json`` keyframe map) — previously a private
    ``_load_index`` cached with ``@lru_cache(maxsize=1)`` keyed only on
    the directory path string, so a *regenerated* index was never picked
    up without a process restart and a stale/corrupt index leaked
    between requests;
  * result-DataFrame → template-dict conversion (``_results_to_dicts``);
  * the text / image search orchestration the route did inline.

Two correctness additions Phase 3c makes on top of the pure extraction
(catalog.py / annotations.py were byte-preserving refactors; this one is
explicitly a validation phase per the plan):

  1. **mtime/size-aware cache invalidation.** The cache is keyed by the
     embeddings + mapping file paths AND their ``(st_mtime_ns, st_size)``
     stat signature. A regenerated index (different size/mtime) is
     re-loaded automatically — no restart, no manual ``cache_clear``.
     Acknowledged blind spot: an index regenerated to a byte-identical
     ``st_size`` AND an identical ``st_mtime_ns`` would not be detected;
     this is practically impossible on a real regeneration (the writer
     touches mtime and the content/size changes), and a content hash is
     the only complete fix — deliberately not done, out of scope for a
     single-worker dev server.

  2. **Index shape validation.** ``CLIPEmbedder.load`` performs NO
     row-count consistency check (see its docstring — it just
     ``np.load`` + ``json.load``). A mapping that declares fewer
     keyframes than the embeddings matrix has rows previously crashed
     ``/api/search`` with a pandas ``IndexError`` (the Phase-2
     ``xfail(strict=True)`` tripwire). Validation lives HERE, in the
     api/services layer, deliberately: it keeps the AI core's contract
     (``embeddings.py``) untouched and does not change embedding/model
     computation or artefact formats — it only refuses to *serve* an
     incoherent index, degrading to a clear "corrupt index" UI state.

Path resolution flows through :class:`FilmContext` (consistent with the
catalog / annotations services). The RAW merged tag index is reused from
``api/services/catalog.load_tag_index`` and passed verbatim to
``SemanticSearch.combined`` — that method self-normalizes (Phase 1c), so
pre-normalizing here would diverge from the characterized contract.

Thread-safety: the module-level cache is guarded by a simple
``threading.Lock`` covering the stat-check + (re)load + store. The lock
is held across the disk load; that is acceptable for the current
single-worker dev server. Request-level concurrency / a job runner is
Phase 4 — this module deliberately does not over-engineer past a
correctness-preserving lock.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from api.services.catalog import (
    derive_fps,
    keyframe_url,  # noqa: F401  — re-exported for api.routes.search
    load_json,
    load_tag_index,
    to_smpte,
)
from api.services.film_context import FilmContext
from cinemateca.retrieval.hybrid import DEFAULT_RRF_K

# Result conversion + Mojica context + films-by-id lookup — relocated to
# cinemateca.search._results and cinemateca.search._lookup (T8).
# Re-exported under the legacy names so external callers
# (``api/routes/search.py``, ``TestResultsToDicts``,
# ``test_multi_film_search.py``) keep working. ``_mojica_search_defaults``
# keeps its leading underscore here (it was private before T8); the new
# home publishes it as ``mojica_search_defaults`` for use within the
# search package.
from cinemateca.search._lookup import (
    build_search_context,  # noqa: F401
    build_search_context_aggregate,  # noqa: F401
    films_by_id_lookup,  # noqa: F401
)
from cinemateca.search._lookup import (
    mojica_search_defaults as _mojica_search_defaults,  # noqa: F401
)
from cinemateca.search._results import results_to_dicts  # noqa: F401

# BM25 loader + lru_cache — relocated to cinemateca.search.bm25.
# Re-exported under the legacy underscored names so external callers
# (``api/routes/search.py``, the existing tests) keep working. The
# module self-registers its cache flusher with
# :func:`cinemateca.search.cache.register_cache_clearer` at import time,
# so the wrapper around ``clear_index_cache`` that lived here under T6
# is gone — calling ``clear_index_cache()`` flushes BM25 transparently.
from cinemateca.search.bm25 import (
    _cached_bm25_index,  # noqa: F401  — legacy name for tests
    _file_stamp,  # noqa: F401  — legacy name for tests
    reindex_bm25,  # noqa: F401  — public P1 verb (T13 wires it into __init__)
)

# CLIP search-index loader + mtime/size cache — relocated to
# cinemateca.search.cache. Re-exported here so the legacy
# ``api.services.search.{IndexStatus, SearchIndex, load_index,
# clear_index_cache}`` import path keeps working for routes and the
# existing test suite (TestLoadIndexValidation, TestCacheInvalidation,
# TestFilmContextWiring). The ``_index_cache`` mapping is re-exported
# under its legacy name as well so tests that poke the dict directly
# (none today, but several reach in via ``cache_mod._index_cache`` in
# T6's own test file) continue to find it via either path.
from cinemateca.search.cache import (
    IndexStatus,  # noqa: F401
    SearchIndex,  # noqa: F401
    _index_cache,  # noqa: F401  — legacy name for tests that poke the dict
    clear_index_cache,  # noqa: F401  — flushes CLIP + BM25 via registered clearers
    load_index,  # noqa: F401
)

# CLIP search verbs — relocated to cinemateca.search.clip (T9). The
# names are re-exported here so external callers
# (``search_service.search_text`` / ``search_service.search_image``)
# and internal callers (``search_hybrid``) keep working unchanged.
from cinemateca.search.clip import (
    search_image,  # noqa: F401
    search_text,  # noqa: F401
)

# Degenerate-tag display filter — relocated to cinemateca.search.display.
# Re-exported under the legacy underscored names so external callers and
# the existing ``TestDegenerateTagFilter`` suite keep working.
from cinemateca.search.display import (
    filter_degenerate_tags as _filter_degenerate_tags,  # noqa: F401
)
from cinemateca.search.display import is_degenerate_tag as _is_degenerate_tag  # noqa: F401

# Upload validation — relocated to cinemateca.search.upload.
# UploadRejected lives in cinemateca.search.types (re-exported here so the
# existing ``api.services.search.UploadRejected`` import path keeps working
# for routes and the legacy ``TestValidateUpload`` suite).
from cinemateca.search.types import UploadRejected  # noqa: F401
from cinemateca.search.upload import (
    ALLOWED_IMAGE_SUFFIXES,  # noqa: F401
    MAX_UPLOAD_BYTES,  # noqa: F401
    validate_upload,  # noqa: F401
)

if TYPE_CHECKING:
    from cinemateca.retrieval.bm25 import BM25Index

logger = logging.getLogger(__name__)


# ── BM25 loader (relocated to cinemateca.search.bm25) ────────────────────────
#
# The core loader, its lru_cache, and the public ``reindex_bm25`` verb
# live in :mod:`cinemateca.search.bm25` (imported at the top of this
# file). This wrapper exists ONLY to resolve the BM25 tunables from the
# FastAPI app config — the core module is config-agnostic so it can be
# imported by tests / scripts without touching ``api.deps``.
def _get_bm25_index_for_ctx(ctx: FilmContext) -> BM25Index:
    """Load + cache the BM25 index for one film, using app-config tunables.

    Resolves ``cfg.search.bm25`` for ``stopwords_lang`` / ``k1`` / ``b``
    via :func:`api.deps.get_config`, then forwards to
    :func:`cinemateca.search.bm25.bm25_index_for_ctx`. ``get_config`` is
    imported lazily so the service module stays loadable without the
    FastAPI app config wired up (matters for unit tests that import
    this module in isolation).
    """
    from api.deps import get_config
    from cinemateca.search.bm25 import bm25_index_for_ctx

    cfg = get_config()
    bm25_cfg = getattr(cfg.search, "bm25", None)
    stopwords_lang = getattr(bm25_cfg, "stopwords_lang", None) if bm25_cfg else None
    k1 = float(getattr(bm25_cfg, "k1", 1.5)) if bm25_cfg else 1.5
    b = float(getattr(bm25_cfg, "b", 0.75)) if bm25_cfg else 0.75
    return bm25_index_for_ctx(
        ctx,
        stopwords_lang=stopwords_lang,
        k1=k1,
        b=b,
    )


# ── Per-film helpers + aggregate search ───────────────────────────────────────

# Canonical filenames for the per-film CLIP index.  These mirror the
# ``config/default.yaml`` → ``embeddings.*`` values and are used as
# defaults when ``cfg.embeddings`` is not present (e.g. in unit tests
# that supply a minimal SimpleNamespace config).
_DEFAULT_EMBEDDINGS_FILENAME = "keyframe_embeddings.npy"
_DEFAULT_MAPPING_FILENAME = "index_mapping.json"


def _get_embedder(cfg: Any) -> object:
    """Return a fresh ``OpenClipEmbedder`` instance.

    Extracted to module scope so unit tests can monkeypatch
    ``api.services.search._get_embedder`` to avoid loading the real CLIP
    model. The ``cfg`` argument is accepted for API consistency and future
    use (e.g. routing to a different backend via cfg) but is currently
    ignored.
    """
    from cinemateca.models.clip.openclip import OpenClipEmbedder

    return OpenClipEmbedder()


def _get_search_index(cfg: Any, slug: str) -> SearchIndex:
    """Return the (cached) :class:`SearchIndex` for the film identified by *slug*.

    Resolves the per-film embeddings directory via
    :meth:`FilmContext.for_film`, then delegates to :func:`load_index`
    with the canonical filenames (read from ``cfg.embeddings`` when
    available, falling back to the constants above for test configs that
    only supply ``paths.library_dir``).
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
        ctx,
        embeddings_filename=embeddings_filename,
        mapping_filename=mapping_filename,
    )


def has_indexed_films(cfg: Any) -> bool:
    """``True`` iff at least one registered film has an OK :class:`SearchIndex`.

    Lets the route distinguish two empty-hit cases:

      * library has no indexed films yet → render "No search index found"
        (user needs to run the embeddings pipeline);
      * library has indexed films but the query produced zero hits above
        ``min_similarity`` → render "No results" (the query simply didn't
        match anything in the corpus — a normal outcome, not a setup error).
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
    from cinemateca.library import scan_library
    from cinemateca.retrieval.hybrid import fuse_rrf
    from cinemateca.scene_ids import normalize_tag_index, scene_id_key

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

    embedder = _get_embedder(cfg)

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
                bm25 = _get_bm25_index_for_ctx(ctx)
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


# ── Hybrid search dispatch (relocated to cinemateca.search.hybrid) ───────────
# CLIP verbs (search_text / search_image) live in cinemateca.search.clip (T9).
# The hybrid dispatcher + its 5 private helpers were extracted to
# cinemateca.search.hybrid in T10. Re-exported here under the legacy name so
# the route layer (``api/routes/search.py``) and the 12 M2 service tests
# (``tests/test_search_hybrid_service.py``) keep working unchanged.
#
# A signature reshape (``query``/``film``/``mode`` form, ``metadata_dir`` in
# place of a pre-loaded ``bm25``) lands in T13 behind the public
# ``cinemateca.search.find()`` verb. Verbatim move first, signature reshape
# behind a stable public surface second.
from cinemateca.search.hybrid import search_hybrid  # noqa: F401,E402

# ``search_image`` relocated to cinemateca.search.clip (T9) — re-exported above.
