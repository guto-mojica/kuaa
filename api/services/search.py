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
    keyframe_url,
    load_json,
    load_tag_index,
    to_smpte,
)
from api.services.film_context import FilmContext
from cinemateca.retrieval.hybrid import DEFAULT_RRF_K

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
    import pandas as pd

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


# ── Result conversion ─────────────────────────────────────────────────────────


def results_to_dicts(
    results_df,
    data_dir: Path,
    meta_by_scene: dict | None = None,
    fps: float = 24.0,
) -> list[dict]:
    """Convert a search result DataFrame to the template's card dicts.

    When ``meta_by_scene`` is supplied (a ``{scene_id: kf_entry}`` dict
    from ``keyframes_metadata.json``), each result row is enriched with a
    SMPTE ``timecode`` field computed from ``start_time_s``. Without it
    the behaviour is byte-equivalent to the prior route implementation.
    """
    out = []
    for row in results_df.to_dict("records"):
        d = {**row, "img_url": keyframe_url(str(row["filepath"]), data_dir)}
        if meta_by_scene is not None:
            meta = meta_by_scene.get(row.get("scene_id"))
            if meta:
                start_s = float(meta.get("start_time_s") or 0.0)
                d["timecode"] = to_smpte(start_s, fps) if start_s > 0 else ""
        out.append(d)
    return out


# ── Search orchestration ──────────────────────────────────────────────────────


def search_text(
    index: SearchIndex,
    query: str,
    tags: list[str],
    tag_index: dict,
    top_k: int,
    min_similarity: float = 0.0,
):
    """Run a text (optionally tag-filtered) semantic search.

    Mirrors the prior route logic exactly: with ``tags`` it calls
    ``SemanticSearch.combined`` passing the RAW merged ``tag_index``
    (``combined`` self-normalizes — Phase 1c contract preserved); without
    tags it calls ``by_text``. Caller (route) is responsible for running
    this in an executor — kept sync here so the service stays
    framework-agnostic and unit-testable without an event loop.

    ``min_similarity`` post-filters the result DataFrame (CLIP returns
    top-K unconditionally, so unrelated queries surface noise scenes;
    the threshold drops anything below the cosine floor). 0.0 disables
    the filter (default for back-compat with unit tests).
    """
    from cinemateca.embeddings import SemanticSearch

    searcher = SemanticSearch(index.embeddings, index.kf_df, index.embedder)
    # The underlying searcher returns the global top-K by similarity; with
    # multiple keyframes per scene that top-K may concentrate inside one
    # scene's keyframe block, starving other scenes. Ask for a wider
    # window (top_k * kf_per_scene) so the post-dedupe top-K still has
    # ``top_k`` distinct scenes to choose from. The wider window only
    # affects ranking, not embedding cost.
    raw_k = top_k * 4  # 4× is the configured ceiling for keyframes_per_scene
    if tags:
        df = searcher.combined(query, tags, tag_index, raw_k)
    else:
        df = searcher.by_text(query, raw_k)
    n_raw = len(df)
    top_raw = float(df["similarity"].iloc[0]) if n_raw and "similarity" in df.columns else 0.0
    if min_similarity > 0.0 and not df.empty and "similarity" in df.columns:
        df = df[df["similarity"] >= min_similarity].reset_index(drop=True)
    # Dedupe by scene_id (Phase-1 density fix). The DataFrame is already
    # ordered by similarity descending, so ``drop_duplicates`` keeps the
    # first occurrence per scene = the best-matching keyframe of that
    # scene. Trim to ``top_k`` AFTER dedup so the UI gets the requested
    # number of *scenes*, not keyframes.
    n_after_floor = len(df)
    if not df.empty and "scene_id" in df.columns:
        df = df.drop_duplicates(subset="scene_id", keep="first").reset_index(drop=True)
    df = df.head(top_k).reset_index(drop=True)
    logger.info(
        "search_text: query=%r top_k=%d tags=%s min_sim=%.3f "
        "raw_hits=%d top_score=%.3f kept_after_floor=%d dedup_kept=%d",
        query,
        top_k,
        tags or None,
        min_similarity,
        n_raw,
        top_raw,
        n_after_floor,
        len(df),
    )
    return df


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

    from cinemateca.retrieval.hybrid import fuse_rrf

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
    import numpy as np

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
    import pandas as pd

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
    import pandas as pd

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
    import pandas as pd

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


def search_image(index: SearchIndex, image_path: Path, top_k: int):
    """Run an image-similarity semantic search (sync; see :func:`search_text`).

    Applies the same scene_id dedupe as :func:`search_text` so the UI
    receives at most one card per scene, displaying the best-matching
    keyframe (rather than three near-duplicate rows from the same shot).
    """
    from cinemateca.embeddings import SemanticSearch

    searcher = SemanticSearch(index.embeddings, index.kf_df, index.embedder)
    df = searcher.by_image(image_path, top_k * 4)
    if not df.empty and "scene_id" in df.columns:
        df = df.drop_duplicates(subset="scene_id", keep="first").reset_index(drop=True)
    return df.head(top_k).reset_index(drop=True)


def _mojica_search_defaults() -> dict:
    """Defaults the Mojica Buscar template (``partials/search.html``)
    needs whenever no actual query has been issued.

    Task 10 introduces a richer template context — query state, view
    toggle, results list, film lookup, highlighted tags — that previous
    tab-renders did not surface. These defaults let the page render the
    initial "type a query to search" empty state with no special casing
    on the template side.

    The per-modality result list is intentionally empty here. Task 11
    fills it with ``.b-card``-shaped dicts produced by the /api/search
    handlers; ``films_by_id`` is populated lazily by callers that have a
    cfg in hand (see :func:`films_by_id_lookup`).
    """
    return {
        "query": "",
        "total": 0,
        "film_count": 0,
        "latency_ms": None,
        "active_mode": "text",
        "active_view": "grid",
        "selected_scene_id": None,
        "results": [],
        "films_by_id": {},
        "highlighted_tags": set(),
    }


def films_by_id_lookup(cfg: Any) -> dict:
    """Return ``{film.slug: film}`` for every registered film.

    Task 11's ``.b-card`` markup looks up ``films_by_id[r.film_slug]`` to
    pull the film title + year onto each result card; the lookup is built
    here so both the per-film and aggregate routes (and the
    ``build_search_context*`` builders) populate the same shape.

    Returns an empty dict when the library directory is absent —
    consistent with :func:`cinemateca.library.scan_library`'s contract.
    Templates should treat the dict as a best-effort lookup
    (``films_by_id.get(slug)``); cards whose ``film_slug`` is missing
    still render with sensible fallbacks.
    """
    from cinemateca.library import scan_library

    library_dir = Path(cfg.paths.library_dir)
    return {film.slug: film for film in scan_library(library_dir)}


def build_search_context(ctx: FilmContext, cfg: Any | None = None) -> dict:
    """Build the per-film search-tab partial context.

    Uses the RAW merged tag index (only its keys feed ``available_tags``
    — identical to the normalized index's keys) and runs them through
    ``_filter_degenerate_tags`` so the pill grid stays clean even when
    ``scene_tags.json`` carries leaked caption fragments.

    Mojica-redesign keys (Task 10) live alongside ``available_tags`` so
    the rewritten template can render the empty state without forcing
    every route to populate them. The ``query`` / ``total`` / ``results``
    defaults are overwritten by ``/api/search`` responses once a query
    fires.

    ``cfg`` is optional for back-compat: when supplied, ``films_by_id``
    is populated via :func:`films_by_id_lookup` so Task-11's ``.b-card``
    template can resolve film titles/years on hits returned by the same
    request. When omitted (legacy callers), ``films_by_id`` stays empty
    and the template falls back to safe-get behaviour.
    """
    tag_index = load_tag_index(ctx.metadata_dir)
    raw_tags = sorted(tag_index.keys()) if tag_index else []
    ctx_dict = _mojica_search_defaults()
    ctx_dict["available_tags"] = _filter_degenerate_tags(raw_tags)
    if cfg is not None:
        ctx_dict["films_by_id"] = films_by_id_lookup(cfg)
    return ctx_dict


def build_search_context_aggregate(cfg: Any) -> dict:
    """Build the aggregate search-tab context (union across all films).

    Mirrors :func:`api.services.catalog.build_scenes_context_aggregate`'s
    tag-union pattern: walks the library registry, unions every film's
    tag-index keys, filters degenerate entries, and returns the same
    ``available_tags`` key the per-film builder exposes — so the
    ``partials/search.html`` template renders identically in either mode.

    Mojica-redesign keys (Task 10) are merged in via
    :func:`_mojica_search_defaults` so the aggregate path and per-film
    path expose the same context shape. ``films_by_id`` is populated
    here so the template's title/year lookup resolves on every card.
    """
    from cinemateca.library import scan_library

    library_dir = Path(cfg.paths.library_dir)
    all_tags: set[str] = set()
    for film in scan_library(library_dir):
        try:
            ctx = FilmContext.for_film(cfg, film.slug)
        except ValueError:
            continue
        tag_index = load_tag_index(ctx.metadata_dir)
        all_tags.update(tag_index.keys())
    ctx_dict = _mojica_search_defaults()
    ctx_dict["available_tags"] = _filter_degenerate_tags(sorted(all_tags))
    ctx_dict["films_by_id"] = films_by_id_lookup(cfg)
    return ctx_dict
