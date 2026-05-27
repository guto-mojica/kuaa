"""Search service — thin HTTP adapter over :mod:`cinemateca.search`.

After P1's deep-modules refactor (T3–T13) the search domain logic lives
in :mod:`cinemateca.search`. This module is the HTTP-adapter surface the
route layer + legacy test suite import through. It re-exports the
symbols the FastAPI app and tests pin against, plus a few small wrappers
that need the FastAPI app config (``_get_embedder`` monkeypatched by
tests; ``_get_search_index`` / ``_get_bm25_index_for_ctx`` resolve
``cfg.embeddings`` / ``cfg.search.bm25`` and forward to
:mod:`cinemateca.search`; ``has_indexed_films`` probes the library;
``dispatch_text_search`` orchestrates the per-film vs aggregate branch).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from api.services.catalog import keyframe_url  # noqa: F401  — used by routes
from cinemateca.library import FilmContext

# Cross-encoder rerank verb (Task 3.1). Aliased so tests can monkeypatch
# ``api.services.search.search_rerank`` without bypassing the wrapper.
from cinemateca.search import rerank as search_rerank  # noqa: F401

# Result conversion + Mojica context + films-by-id lookup (T8). T15
# adds ``enrich_hits_with_film_metadata`` re-export so the slim route
# layer doesn't import ``cinemateca.search._lookup`` directly (keeps
# the routes-not-direct-core import-linter contract clean).
from cinemateca.search._lookup import (
    build_search_context,  # noqa: F401
    build_search_context_aggregate,  # noqa: F401
    enrich_hits_with_film_metadata,  # noqa: F401
    films_by_id_lookup,  # noqa: F401
)
from cinemateca.search._results import results_to_dicts  # noqa: F401

# Aggregate cross-film search (T11) — still reads ``_get_embedder`` and
# ``_get_search_index`` off this module via lazy attribute access, so the
# monkeypatches on ``api.services.search._get_*`` keep hitting the call path.
# T15 adds ``aggregate_hits_to_template_dicts`` re-export (same rationale
# as the ``enrich_hits_with_film_metadata`` re-export above).
from cinemateca.search.aggregate import (  # noqa: F401
    _get_embedder,
    _get_search_index,
    aggregate_hits_to_template_dicts,
    aggregate_search,
    has_indexed_films,
)

# BM25 loader + lru_cache (T7) — module self-registers its cache flusher
# with cinemateca.search.cache so ``clear_index_cache()`` flushes BM25.
from cinemateca.search.bm25 import (
    _cached_bm25_index,  # noqa: F401  — legacy name for tests
    _file_stamp,  # noqa: F401  — legacy name for tests
)

# CLIP search-index loader + mtime/size cache (T6).
from cinemateca.search.cache import (
    IndexStatus,  # noqa: F401
    SearchIndex,  # noqa: F401
    clear_index_cache,  # noqa: F401  — flushes CLIP + BM25
    load_index,  # noqa: F401
)

# CLIP search verbs (T9).
from cinemateca.search.clip import (
    search_image,  # noqa: F401
    search_text,  # noqa: F401
)

# Degenerate-tag display filter (T4).
from cinemateca.search.display import (
    filter_degenerate_tags as _filter_degenerate_tags,  # noqa: F401
)

# Hybrid dispatch (T10). T15 adds ``resolve_retriever_args`` so the
# slim route imports HTTP-input normalisation from this layer rather
# than reaching into ``cinemateca.search.hybrid`` directly.
from cinemateca.search.hybrid import (  # noqa: F401
    resolve_retriever_args,
    search_hybrid,
)

# Upload validation (T5). UploadRejected re-exported for the legacy
# ``api.services.search.UploadRejected`` import path used by routes + tests.
from cinemateca.search.types import (  # noqa: F401
    Hit,
    Query,
    SearchMode,
    SearchResult,
    UploadRejected,
)
from cinemateca.search.upload import (
    MAX_UPLOAD_BYTES,  # noqa: F401
    validate_upload,  # noqa: F401
)

if TYPE_CHECKING:
    from cinemateca.retrieval.bm25 import BM25Index

logger = logging.getLogger(__name__)


def _get_bm25_index_for_ctx(ctx: FilmContext) -> BM25Index:
    """Load + cache the BM25 index for one film. Resolves ``cfg.search.bm25``
    tunables (``stopwords_lang`` / ``k1`` / ``b``) via lazy ``get_config``
    so this module stays loadable without the FastAPI app wired up.
    """
    from api.deps import get_config
    from cinemateca.search.bm25 import bm25_index_for_ctx

    cfg = get_config()
    bm25_cfg = getattr(cfg.search, "bm25", None)
    stopwords_lang = getattr(bm25_cfg, "stopwords_lang", None) if bm25_cfg else None
    k1 = float(getattr(bm25_cfg, "k1", 1.5)) if bm25_cfg else 1.5
    b = float(getattr(bm25_cfg, "b", 0.75)) if bm25_cfg else 0.75
    return bm25_index_for_ctx(ctx, stopwords_lang=stopwords_lang, k1=k1, b=b)


# ── Cross-encoder reranker (Task 3.2) ────────────────────────────────────────
# ``apply_reranker`` is the cfg-aware wrapper over the pure
# :func:`cinemateca.search.rerank` verb (imported above as ``search_rerank``).
# Callers invoke it once at the OUTERMOST :class:`SearchResult` boundary so we
# never double-rerank when an aggregate path composes sub-results internally.
# The reader helper centralises the ``retrieval.reranker.*`` defaults so a
# cfg without the block (or without ``retrieval`` at all) doesn't raise.


def _reranker_settings(cfg: Any, enabled_override: bool | None = None) -> tuple[bool, str, int]:
    """Read ``retrieval.reranker.{enabled,model,top_k_in}`` with defaults.

    Defaults: ``enabled=True``, ``model='default'``, ``top_k_in=20``. A cfg
    missing ``retrieval`` or ``retrieval.reranker`` falls back to all-defaults
    silently — callers should never see an ``AttributeError`` from a partial
    config.
    """
    rr = getattr(getattr(cfg, "retrieval", None), "reranker", None)
    if rr is None:
        enabled, model, top_k_in = True, "default", 20
    else:
        enabled = bool(getattr(rr, "enabled", True))
        model = str(getattr(rr, "model", "default"))
        top_k_in = int(getattr(rr, "top_k_in", 20))
    if enabled_override is not None:
        enabled = bool(enabled_override)
    return (enabled, model, top_k_in)


def apply_reranker(
    result: SearchResult, *, cfg: Any, enabled_override: bool | None = None
) -> SearchResult:
    """Apply the cross-encoder reranker to a :class:`SearchResult`.

    Reads ``retrieval.reranker.*`` from ``cfg``; ``enabled_override`` lets
    the request-level ``?reranker_enabled=`` toggle opt in/out without
    mutating global config. Safe to call unconditionally at the outermost
    boundary of any retriever path that produces a :class:`SearchResult`.
    Tests can stub the underlying verb with
    ``monkeypatch.setattr(svc, "search_rerank", ...)``.
    """
    enabled, model, top_k_in = _reranker_settings(cfg, enabled_override)
    if not enabled:
        return result
    return search_rerank(result, model=model, top_k_in=top_k_in)


def _result_key(row: dict[str, Any]) -> tuple[str, int]:
    return (str(row.get("film_slug") or ""), int(row.get("scene_id") or 0))


def _row_score(row: dict[str, Any]) -> float:
    raw = row.get("similarity", row.get("score", 0.0))
    try:
        return float(raw or 0.0)
    except (TypeError, ValueError):
        return 0.0


def rerank_template_results(
    results: list[dict[str, Any]],
    *,
    cfg: Any,
    query: str,
    mode: str = "hybrid",
    enabled: bool | None = None,
) -> list[dict[str, Any]]:
    """Apply the text reranker to enriched template-result dicts.

    Current HTTP dispatchers still produce DataFrames / ``list[dict]`` before
    route-level enrichment adds descriptions and tags. The cross-encoder needs
    those descriptions, so adapt the final card dicts into ``SearchResult``,
    rerank once, then return the same dict shape in reranked order.
    """
    if enabled is False or not results:
        return results

    originals: dict[tuple[str, int], dict[str, Any]] = {}
    hits: list[Hit] = []
    for row in results:
        try:
            sid = int(row.get("scene_id") or 0)
        except (TypeError, ValueError):
            continue
        key = (str(row.get("film_slug") or ""), sid)
        originals[key] = row
        tags = row.get("tags") or []
        hits.append(
            Hit(
                scene_id=sid,
                score=_row_score(row),
                keyframe_path=str(row.get("keyframe_path") or row.get("filepath") or ""),
                film_slug=key[0] or None,
                film_title=row.get("film_title"),
                timecode=str(row.get("timecode") or ""),
                description=str(row.get("description") or ""),
                tags=list(tags) if isinstance(tags, list) else [],
            )
        )
    if not hits:
        return results

    search_mode = cast(SearchMode, mode if mode in {"clip", "bm25", "hybrid"} else "hybrid")
    search_result = SearchResult(
        hits=hits,
        mode=search_mode,
        weights=None,
        query=Query.text_query(query),
        no_index=False,
    )
    try:
        reranked = apply_reranker(search_result, cfg=cfg, enabled_override=enabled)
    except Exception as exc:
        logger.warning("reranker failed; leaving text results unchanged: %s", exc)
        return results

    ordered: list[dict[str, Any]] = []
    used: set[tuple[str, int]] = set()
    for hit in reranked.hits:
        key = (hit.film_slug or "", hit.scene_id)
        original = originals.get(key)
        if original is None:
            continue
        row = dict(original)
        if hit.rerank_score is not None:
            row["rerank_score"] = hit.rerank_score
        ordered.append(row)
        used.add(key)

    if not ordered:
        return results
    # If the configured top_k_in is lower than the requested top-k, preserve
    # unscored tail results after the reranked head instead of making cards
    # disappear merely because reranking is enabled.
    ordered.extend(row for row in results if _result_key(row) not in used)
    return ordered


def dispatch_audio_search(
    cfg: Any,
    ctx: Any | None,
    q: str,
    top_k: int,
) -> tuple[list[dict], bool]:
    """Run audio (CLAP) search; return ``(hits, no_index)``.

    ``ctx=None`` → cross-film: walk every registered film, read its
    per-film CLAP index, run :func:`search_audio` against the same
    query vector, concatenate results, and take the global top-``top_k``
    by raw cosine. CLAP vectors are L2-normalised AND share a single
    joint text+audio space, so cosines are cross-film-comparable — no
    fusion / RRF reshape needed for parity with the CLIP path.

    ``ctx`` given → per-film: load ``<film_dir>/audio/`` only.

    ``no_index=True`` is returned when:
      * per-film: that film has no CLAP index on disk;
      * aggregate: no registered film has a CLAP index.

    The route renders the no-index empty state in both cases, same
    contract as the text path. The embedder is instantiated exactly
    once per dispatch (so the aggregate path doesn't reload CLAP per
    film); the call site monkeypatches
    ``cinemateca.models.registry.get_audio_embedder`` to skip the
    real model load in tests.
    """
    from cinemateca.library import scan_library
    from cinemateca.models.registry import get_audio_embedder
    from cinemateca.search.audio import load_audio_index, search_audio

    if ctx is not None:
        audio_dir = Path(ctx.metadata_dir).parent / "audio"
        index = load_audio_index(audio_dir)
        if index is None:
            return [], True
        embedder = get_audio_embedder(cfg, device=None)
        hits = search_audio(index, embedder, q, top_k=top_k)
        for h in hits:
            h["film_slug"] = ctx.slug
        return hits, False

    # Aggregate. Walk the registry, skip films without a CLAP index.
    library_dir = Path(cfg.paths.library_dir)
    films = list(scan_library(library_dir))
    if not films:
        return [], True
    embedder = None
    all_hits: list[dict] = []
    any_index = False
    for film in films:
        film_audio_dir = library_dir / film.slug / "audio"
        idx = load_audio_index(film_audio_dir)
        if idx is None:
            continue
        any_index = True
        if embedder is None:
            # Lazy-load the embedder ONCE after we know at least one
            # film has a CLAP index — keeps the empty-library / no-CLAP
            # case from paying the CLAP load cost.
            embedder = get_audio_embedder(cfg, device=None)
        film_hits = search_audio(idx, embedder, q, top_k=top_k)
        for h in film_hits:
            h["film_slug"] = film.slug
            h["film_title"] = film.title
            all_hits.append(h)
    if not any_index:
        return [], True
    all_hits.sort(key=lambda r: r["score"], reverse=True)
    return all_hits[:top_k], False


# ── Fusion helpers ───────────────────────────────────────────────────────────
# ``_NullEncoder`` is handed to :func:`search_fusion` for whichever modality
# isn't present for a given film. Hoisted to module scope so the class object
# is allocated once at import time instead of once per per-film call.


class _NullEncoder:
    """Stub text encoder used when a modality is absent for a given film.

    Paired with a ``(0, 1)`` stub embedding matrix so the dim guard in
    :func:`cinemateca.search.fusion.search_fusion` passes; cosine ops over
    the 0-row matrix produce an empty per-modality top-k and the modality
    contributes no rows to the union (no active penalty, no encode cost).
    """

    def encode_text(self, text: str) -> Any:  # pragma: no cover - trivial
        import numpy as np

        return np.zeros(1, dtype="float32")


def _normalise_clip_mapping(raw: Any) -> list[dict]:
    """Coerce a CLIP ``index_mapping.json`` payload to ``list[dict]`` with
    at least the ``scene_id`` key. Handles both on-disk shapes:

    * **dict-of-parallel-arrays** (current OpenClip / SigLIP2 writer):
      ``{"scene_ids": [...], "keyframe_paths": [...], ...}`` — keys may
      include any subset of the parallel arrays.
    * **list-of-dicts** (legacy / synthetic test fixtures):
      ``[{"scene_id": int, ...}, ...]`` — already row-shaped.

    Mirrors :func:`cinemateca.search.audio.load_audio_index`'s CLAP
    normaliser. Extra fields are preserved on the list-of-dicts side
    but ignored from the dict-of-arrays side (only ``scene_ids`` is
    load-bearing for fusion's score lookup).
    """
    if isinstance(raw, dict) and "scene_ids" in raw:
        sids = raw["scene_ids"]
        return [{"scene_id": int(sids[i])} for i in range(len(sids))]
    if isinstance(raw, list):
        # Coerce defensively — scene_id may be float/str in legacy files.
        return [{"scene_id": int(m["scene_id"])} for m in raw]
    raise ValueError(
        "Unrecognised CLIP index_mapping shape: expected dict with "
        "'scene_ids' key or list of dicts."
    )


def _fusion_per_film_by_paths(
    *,
    cfg: Any,
    slug: str,
    embeddings_dir: Path,
    audio_dir: Path,
    q: str,
    top_k: int,
    visual_weight: float,
    k_each: int,
    clip_embedder: Any | None,
    clap_embedder: Any | None,
) -> tuple[list[dict], bool, Any | None, Any | None]:
    """Run fusion for one film, addressed purely by paths (no FilmContext).

    Returns ``(hits, no_index, clip_embedder, clap_embedder)`` so the
    aggregate caller can thread + reuse a single embedder instance across
    films. ``no_index=True`` when neither CLIP nor CLAP indices exist.
    """
    import json as _json

    import numpy as np

    from cinemateca.models.registry import get_audio_embedder, get_image_embedder
    from cinemateca.search.audio import load_audio_index
    from cinemateca.search.fusion import FusionConfig, search_fusion

    clip_emb_path = embeddings_dir / "keyframe_embeddings.npy"
    clip_map_path = embeddings_dir / "index_mapping.json"

    has_clip = clip_emb_path.exists() and clip_map_path.exists()
    audio_idx = load_audio_index(audio_dir)
    has_clap = audio_idx is not None

    if not has_clip and not has_clap:
        return [], True, clip_embedder, clap_embedder

    # Lazy-load embedders only when we actually need them. The aggregate
    # path threads instances back out so subsequent films reuse them.
    if has_clip and clip_embedder is None:
        clip_embedder = get_image_embedder(cfg, device=None)
    if has_clap and clap_embedder is None:
        clap_embedder = get_audio_embedder(cfg, device=None)

    if has_clip:
        clip_emb = np.load(clip_emb_path).astype("float32", copy=False)
        clip_mapping = _normalise_clip_mapping(_json.loads(clip_map_path.read_text()))
    else:
        # Build a (0, 1) stub matrix + a 1-dim NullEncoder so the dim guard in
        # search_fusion passes; the modality contributes no rows to the union.
        clip_emb = np.zeros((0, 1), dtype="float32")
        clip_mapping = []

    if has_clap:
        assert audio_idx is not None  # narrow for mypy
        clap_emb = audio_idx.embeddings
        clap_mapping = [{"scene_id": int(m["scene_id"])} for m in audio_idx.mapping]
    else:
        # Same (0, 1) stub matrix + NullEncoder pattern as the CLIP-missing
        # branch above; passes the dim guard, contributes no rows.
        clap_emb = np.zeros((0, 1), dtype="float32")
        clap_mapping = []

    clip_for_call = clip_embedder if has_clip else _NullEncoder()
    clap_for_call = clap_embedder if has_clap else _NullEncoder()

    hits = search_fusion(
        clip_emb=clip_emb,
        clap_emb=clap_emb,
        clip_mapping=clip_mapping,
        clap_mapping=clap_mapping,
        query_text=q,
        clip_embedder=clip_for_call,
        clap_embedder=clap_for_call,
        cfg=FusionConfig(visual_weight=visual_weight, k_each=k_each, k_final=top_k),
    )
    for h in hits:
        h["film_slug"] = slug
    return hits, False, clip_embedder, clap_embedder


def dispatch_fusion_search(
    cfg: Any,
    ctx: Any | None,
    q: str,
    top_k: int,
    *,
    visual_weight: float = 0.5,
    k_each: int = 50,
) -> tuple[list[dict], bool]:
    """Cross-modal CLIP × CLAP fusion search; returns ``(hits, no_index)``.

    Mirrors :func:`dispatch_audio_search` deliberately — same dispatch
    shape (per-film vs aggregate), same ``no_index`` semantics, same
    lazy-imports + at-most-once embedder load on the aggregate path.

    Behaviour:

    * ``ctx`` given → per-film. Loads ``<film>/embeddings/`` (CLIP) and
      ``<film>/audio/`` (CLAP). If both are missing → ``([], True)``.
      If only one is present, the missing modality contributes
      ``0.0`` to every fused score (the verb already handles this via
      zero-row stubs, so missing modalities don't actively penalise).
    * ``ctx=None`` → cross-film. Walks
      :func:`scan_library`, runs the per-film fusion logic for each
      film, and takes the global top-``top_k`` by fused score. Films
      with neither index are silently skipped. If NO film has any
      index → ``([], True)``. The aggregate path derives per-film
      paths directly from ``cfg.paths.library_dir + film.slug`` and
      deliberately avoids :meth:`FilmContext.for_film`, which would
      mkdir per-film subdirs on every query.

    Hits carry ``film_slug`` (always) and ``film_title`` (aggregate
    path only — the per-film caller already knows the title). Per-hit
    keys are ``scene_id`` / ``score`` / ``clip_score`` / ``clap_score``
    from :func:`search_fusion`.

    Test seam: monkeypatch
    ``cinemateca.models.registry.{get_image_embedder, get_audio_embedder}``
    to avoid loading real CLIP/CLAP weights. ``cfg`` is consulted only
    for ``cfg.paths.library_dir`` (aggregate path) and is forwarded to
    the registry factories — service-layer config-shape decisions
    (e.g. ``cfg.retrieval.fusion.visual_weight``) live in the route
    layer (Task 3.1) and arrive here as kwargs.
    """
    from cinemateca.library import scan_library

    if ctx is not None:
        embeddings_dir = ctx.embeddings_dir
        audio_dir = Path(ctx.metadata_dir).parent / "audio"
        hits, no_index, _, _ = _fusion_per_film_by_paths(
            cfg=cfg,
            slug=ctx.slug,
            embeddings_dir=embeddings_dir,
            audio_dir=audio_dir,
            q=q,
            top_k=top_k,
            visual_weight=visual_weight,
            k_each=k_each,
            clip_embedder=None,
            clap_embedder=None,
        )
        return hits, no_index

    # Aggregate. Walk the registry; skip films with neither modality.
    # Derive per-film paths directly — do NOT call FilmContext.for_film
    # here, it mkdir's raw/metadata/frames/embeddings on every call (same
    # avoidance as dispatch_audio_search above).
    library_dir = Path(cfg.paths.library_dir)
    films = list(scan_library(library_dir))
    if not films:
        return [], True
    clip_embedder: Any | None = None
    clap_embedder: Any | None = None
    all_hits: list[dict] = []
    any_film = False
    for film in films:
        film_embeddings_dir = library_dir / film.slug / "embeddings"
        film_audio_dir = library_dir / film.slug / "audio"
        film_hits, film_no_index, clip_embedder, clap_embedder = _fusion_per_film_by_paths(
            cfg=cfg,
            slug=film.slug,
            embeddings_dir=film_embeddings_dir,
            audio_dir=film_audio_dir,
            q=q,
            top_k=top_k,
            visual_weight=visual_weight,
            k_each=k_each,
            clip_embedder=clip_embedder,
            clap_embedder=clap_embedder,
        )
        if film_no_index:
            continue
        any_film = True
        for h in film_hits:
            h["film_title"] = film.title
            all_hits.append(h)
    if not any_film:
        return [], True
    all_hits.sort(key=lambda r: r["score"], reverse=True)
    return all_hits[:top_k], False


def audio_hits_to_template_dicts(
    cfg: Any, hits: list[dict], *, per_film_slug: str | None = None
) -> list[dict]:
    """Convert :func:`dispatch_audio_search` raw hits to template-card dicts.

    Mirrors :func:`aggregate_hits_to_template_dicts` (the CLIP-aggregate
    converter): aliases ``score → similarity`` so the same
    ``partials/search_results.html`` renders the CLAP path, resolves
    a per-scene ``keyframe_path`` + ``timecode`` from each film's
    ``keyframes_metadata.json`` so the card shows the visual proxy for
    the matched audio segment, and resolves ``img_url`` via
    :func:`cinemateca.library.keyframe_url`.

    Per-scene metadata lookup is memoised per slug — a 100-row result
    list reads each film's ``keyframes_metadata.json`` at most once.
    """
    from cinemateca.library import (
        FilmContext,
        derive_fps,
        load_json,
        to_smpte,
    )

    data_dir = Path(cfg.paths.data_dir).resolve()
    kf_cache: dict[str, tuple[dict, float]] = {}

    def _kf_for(slug: str) -> tuple[dict, float]:
        if slug in kf_cache:
            return kf_cache[slug]
        try:
            ctx = FilmContext.for_film(cfg, slug)
        except ValueError:
            kf_cache[slug] = ({}, 24.0)
            return kf_cache[slug]
        kf_meta = load_json(ctx.metadata_dir / "keyframes_metadata.json") or []
        by_scene = {int(e["scene_id"]): e for e in kf_meta if "scene_id" in e}
        kf_cache[slug] = (by_scene, derive_fps(kf_meta))
        return kf_cache[slug]

    out: list[dict] = []
    for h in hits:
        slug = h.get("film_slug") or per_film_slug or ""
        sid = int(h["scene_id"])
        by_scene, fps = _kf_for(slug) if slug else ({}, 24.0)
        meta = by_scene.get(sid) or {}
        kf_path = meta.get("filepath", "") or meta.get("keyframe_path", "") or ""
        start_s = float(meta.get("start_time_s") or 0.0)
        out.append(
            {
                "film_slug": slug,
                "scene_id": sid,
                "similarity": float(h["score"]),
                "img_url": keyframe_url(kf_path, data_dir) if kf_path else None,
                "timecode": to_smpte(start_s, fps) if start_s > 0 else "",
            }
        )
    return out


def dispatch_text_search(
    cfg: Any,
    ctx: Any | None,
    q: str,
    tags: list[str],
    top_k: int,
    min_sim: float,
    retriever: str,
    sw: float,
    bw: float,
    rrf_k: int,
) -> tuple[Any, bool]:
    """Run text search; return ``(payload, no_index)``.

    ``ctx=None`` → cross-film aggregate (``payload`` = list of hit dicts).
    ``ctx`` given → per-film (``payload`` = a DataFrame). ``no_index=True``
    signals the route should render the no-index empty state instead of a
    results list. Pulls the BM25 index + tag index lazily so the per-film
    fast path doesn't read disk when ``retriever == "clip"`` and ``tags``
    is empty (preserving the legacy behaviour).

    When ``cfg.search.rerank_enabled`` is true the per-film first stage
    retrieves ``reranker.top_k`` candidates (default 50), then
    ``rerank_dataframe`` re-scores them with a cross-encoder and trims to
    the original ``top_k``. Aggregate mode is not reranked (each per-film
    result is already trimmed before aggregation).
    """
    from api.services.catalog import load_tag_index

    rerank_enabled = bool(getattr(cfg.search, "rerank_enabled", False))
    reranker_cfg = getattr(cfg.search, "reranker", None)
    candidate_k = int(getattr(reranker_cfg, "top_k", 50)) if reranker_cfg else 50

    if ctx is None:
        try:
            hits = aggregate_search(
                cfg,
                query=q,
                modality="text",
                top_k=top_k,
                tags=tags,
                min_similarity=min_sim,
                retriever_mode=retriever,
                sem_w=sw,
                bm25_w=bw,
                rrf_k=rrf_k,
            )
        except NotImplementedError:
            return [], True
        if not hits and not has_indexed_films(cfg):
            return [], True
        return hits, False

    index = load_index(
        ctx,
        mapping_filename=cfg.embeddings.mapping_filename,
        embeddings_filename=cfg.embeddings.filename,
        cfg=cfg,
    )
    if not index.ok:
        return None, True
    tag_index = load_tag_index(ctx.metadata_dir) if tags else {}

    # Widen first-stage retrieval when reranking so the cross-encoder has
    # enough candidates to reshuffle. Trimming to ``top_k`` happens after.
    first_k = max(top_k, candidate_k) if rerank_enabled and q else top_k

    if retriever == "clip":
        result_df = search_text(index, q, tags, tag_index, first_k, min_sim)
    else:
        bm25 = _get_bm25_index_for_ctx(ctx)
        result_df = search_hybrid(
            index,
            bm25=bm25,
            query=q,
            tags=tags,
            tag_index=tag_index,
            top_k=first_k,
            min_similarity=min_sim,
            retriever_mode=retriever,
            sem_w=sw,
            bm25_w=bw,
            rrf_k=rrf_k,
        )

    return result_df, False
