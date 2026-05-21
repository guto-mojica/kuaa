"""Search tab routes — text and image semantic search via CLIP.

Thin HTTP layer: request parse + executor offload + template render.
All index loading (now mtime/size-cache-invalidated), shape validation,
upload guards and result conversion live in
``api/services/search.py`` (Phase 3c). Path resolution flows through
:class:`FilmContext` (consistent with scenes / annotate).

T9: ``/api/search`` accepts an optional ``?film=<slug>`` query parameter.
``slug=None`` → aggregate-search across all films (``aggregate_search``);
``slug given`` → per-film index search (``_get_search_index`` + existing
text-search path).  Image search is per-film only (aggregate image search
lands in a later plan).
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Query, Request, UploadFile
from fastapi.responses import HTMLResponse

from api.deps import film_slug_query, get_config, make_ctx
from api.services import search as search_service
from api.services.catalog import derive_fps, load_json, load_tag_index
from api.services.film_context import FilmContext
from api.templates import templates

logger = logging.getLogger(__name__)
router = APIRouter()


# Re-exported so api/server.py's tab-context map (``"search":
# search.build_search_context``) keeps working without churn; the
# implementation now lives in the service.
def build_search_context(slug: str | None = None) -> dict:
    """Build the search-tab partial context (delegates to the service).

    With ``slug`` set, the per-film tag vocabulary is used; with
    ``slug=None`` the aggregate (cross-film) union is used so the tag
    pills shown match the search scope. Both paths drop degenerate-looking
    entries via ``_filter_degenerate_tags`` (display-only).

    ``cfg`` is threaded into the per-film service builder so Task 11's
    ``.b-card`` template gets a populated ``films_by_id`` lookup in
    either mode (matching the aggregate builder's behaviour).
    """
    cfg = get_config()
    if slug is not None:
        return search_service.build_search_context(FilmContext.for_film(cfg, slug), cfg)
    return search_service.build_search_context_aggregate(cfg)


def _kf_meta(ctx: FilmContext) -> tuple[dict, float]:
    """Return ``(meta_by_scene, fps)`` from ``keyframes_metadata.json``."""
    kf_meta = load_json(ctx.metadata_dir / "keyframes_metadata.json") or []
    fps = derive_fps(kf_meta)
    meta_by_scene = {e["scene_id"]: e for e in kf_meta if "scene_id" in e}
    return meta_by_scene, fps


def _no_index_response(request: Request) -> HTMLResponse:
    """Render the graceful no-index / corrupt-index results state."""
    return templates.TemplateResponse(
        request,
        "partials/search_results.html",
        make_ctx(request, results=[], no_index=True, films_by_id={}),
    )


def _enrich_with_film_metadata(
    cfg, results: list[dict], *, per_film_slug: str | None = None
) -> list[dict]:
    """Decorate result dicts with ``film_slug``, ``description``, ``tags``.

    The Task-11 ``.b-card`` template reads ``r.film_slug``,
    ``r.description``, ``r.tags`` and ``r.pin_count``. Aggregate hits
    already carry ``film_slug`` (set by ``aggregate_search``); per-film
    hits don't, so we inject ``per_film_slug`` there. Description + tag
    fields are looked up from each film's metadata directory once
    (memoised by slug) so a 100-row result list reads each film's
    descriptions / tag_index at most once.

    Missing data is benign — ``description`` falls back to ``""`` and
    ``tags`` to ``[]``, both of which the template handles. ``pin_count``
    is always 0 today; the persistence layer for pins lands later in M1.
    """
    desc_cache: dict[str, dict[int, str]] = {}
    tag_cache: dict[str, dict[int, list[str]]] = {}

    def _load_film_meta(slug: str) -> tuple[dict[int, str], dict[int, list[str]]]:
        if slug in desc_cache:
            return desc_cache[slug], tag_cache[slug]
        try:
            ctx = FilmContext.for_film(cfg, slug)
        except ValueError:
            desc_cache[slug] = {}
            tag_cache[slug] = {}
            return desc_cache[slug], tag_cache[slug]
        descs_raw = load_json(ctx.metadata_dir / "scene_descriptions.json") or []
        descs: dict[int, str] = {}
        if isinstance(descs_raw, list):
            for entry in descs_raw:
                sid = entry.get("scene_id")
                if sid is None:
                    continue
                try:
                    descs[int(sid)] = str(entry.get("description") or "")
                except (TypeError, ValueError):
                    continue
        # Invert the merged tag_index into {scene_id: [tag, …]} so we
        # can pull per-scene tag pills in O(1). load_tag_index merges
        # LLM + manual; the v0.3 catalog template already trusts this
        # merged view to be the "displayable" tag set.
        merged = load_tag_index(ctx.metadata_dir) or {}
        per_scene: dict[int, list[str]] = {}
        for tag, sids in merged.items():
            if not isinstance(sids, (list, set, tuple)):
                continue
            for sid in sids:
                try:
                    key = int(sid)
                except (TypeError, ValueError):
                    continue
                per_scene.setdefault(key, []).append(tag)
        desc_cache[slug] = descs
        tag_cache[slug] = per_scene
        return descs, per_scene

    enriched: list[dict] = []
    for r in results:
        r = dict(r)  # don't mutate the caller's dicts
        slug = r.get("film_slug") or per_film_slug
        if slug and "film_slug" not in r:
            r["film_slug"] = slug
        sid_raw = r.get("scene_id")
        sid = None
        if sid_raw is not None:
            try:
                sid = int(sid_raw)
            except (TypeError, ValueError):
                sid = None
        if slug and sid is not None:
            descs, tags_by_scene = _load_film_meta(slug)
            r.setdefault("description", descs.get(sid, ""))
            r.setdefault("tags", tags_by_scene.get(sid, []))
        else:
            r.setdefault("description", "")
            r.setdefault("tags", [])
        r.setdefault("pin_count", 0)
        enriched.append(r)
    return enriched


@router.get("/tab/search", response_class=HTMLResponse)
async def tab_search(
    request: Request,
    slug: str | None = Depends(film_slug_query),
) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "partials/search.html",
        make_ctx(request, current_slug=slug, **build_search_context(slug)),
    )


@router.get("/api/search", response_class=HTMLResponse)
async def api_search(
    request: Request,
    q: str = "",
    tags: list[str] = Query(default=[]),
    top_k: int = 8,
    slug: str | None = Depends(film_slug_query),
) -> HTMLResponse:
    q = q.strip()
    if len(q) < 2:
        return HTMLResponse("")

    cfg = get_config()
    min_sim = float(getattr(cfg.embeddings, "min_similarity", 0.0) or 0.0)
    logger.info(
        "api_search: query=%r slug=%s tags=%s top_k=%d min_sim=%.3f",
        q,
        slug or "(aggregate)",
        list(tags) or None,
        top_k,
        min_sim,
    )

    if slug is None:
        # Aggregate search: run per-film text search across all registered
        # films and merge results by cosine score.  Tags are not yet
        # supported in aggregate mode (per-film tag post-filtering lands in
        # a later plan); fall back to no-index response when there are no
        # indexed films.
        loop = asyncio.get_event_loop()
        try:
            hits = await loop.run_in_executor(
                None,
                lambda: search_service.aggregate_search(
                    cfg,
                    query=q,
                    modality="text",
                    top_k=top_k,
                    tags=tags,
                    min_similarity=min_sim,
                ),
            )
        except NotImplementedError:
            return _no_index_response(request)
        if not hits:
            # Empty hits is ambiguous: either no films are indexed (the
            # user needs to run the pipeline), or every per-film hit was
            # below ``min_similarity`` (the query simply matched nothing).
            # ``has_indexed_films`` distinguishes them so the message
            # matches the real cause.
            if not search_service.has_indexed_films(cfg):
                return _no_index_response(request)
            return templates.TemplateResponse(
                request,
                "partials/search_results.html",
                make_ctx(
                    request,
                    current_slug=slug,
                    results=[],
                    no_index=False,
                    query=q,
                    films_by_id=search_service.films_by_id_lookup(cfg),
                    highlighted_tags=set(tags),
                ),
            )
        # Convert aggregate hit dicts → template-compatible result dicts.
        # aggregate_search returns plain dicts (not DataFrames), so
        # results_to_dicts is not applicable; build the img_url here.
        # data_dir must be the media-mount root (cfg.paths.data_dir), NOT
        # library_dir — otherwise keyframe_url's relative_to() check fails
        # for filepaths stored under data/frames/... or data/library/<slug>/...
        # and the template gets img_url=None for every row.
        data_dir = Path(cfg.paths.data_dir).resolve()
        results = [
            {
                "film_slug": h["film_slug"],
                "film_title": h["film_title"],
                "scene_id": h["scene_id"],
                # Template uses ``r.similarity``; aggregate_search emits
                # ``score`` (cosine over the per-film index). Alias here so
                # the same partial works for per-film and aggregate paths.
                "similarity": h["score"],
                "img_url": search_service.keyframe_url(h["keyframe_path"], data_dir),
                # ``aggregate_search`` loads each film's keyframes_metadata
                # once and computes SMPTE per hit; empty string when the
                # film has no metadata (template hides the span).
                "timecode": h["timecode"],
            }
            for h in hits
        ]
        results = _enrich_with_film_metadata(cfg, results)
        return templates.TemplateResponse(
            request,
            "partials/search_results.html",
            make_ctx(
                request,
                current_slug=slug,
                results=results,
                no_index=False,
                query=q,
                films_by_id=search_service.films_by_id_lookup(cfg),
                highlighted_tags=set(tags),
            ),
        )

    # Per-film search: slug is guaranteed non-None here (aggregate path
    # returned above), so resolve the per-film context directly.
    ctx = FilmContext.for_film(cfg, slug)
    index = search_service.load_index(
        ctx,
        mapping_filename=cfg.embeddings.mapping_filename,
        embeddings_filename=cfg.embeddings.filename,
    )
    if not index.ok:
        return _no_index_response(request)

    tag_index = load_tag_index(ctx.metadata_dir) if tags else {}
    loop = asyncio.get_event_loop()
    results_df = await loop.run_in_executor(
        None, search_service.search_text, index, q, tags, tag_index, top_k, min_sim
    )

    meta_by_scene, fps = _kf_meta(ctx)
    results = _enrich_with_film_metadata(
        cfg,
        search_service.results_to_dicts(results_df, ctx.data_dir, meta_by_scene, fps),
        per_film_slug=slug,
    )
    return templates.TemplateResponse(
        request,
        "partials/search_results.html",
        make_ctx(
            request,
            current_slug=slug,
            results=results,
            no_index=False,
            query=q,
            films_by_id=search_service.films_by_id_lookup(cfg),
            highlighted_tags=set(tags),
        ),
    )


@router.post("/api/search/image", response_class=HTMLResponse)
async def api_search_image(
    request: Request,
    file: UploadFile = File(...),
    top_k: int = 8,
    slug: str | None = Depends(film_slug_query),
) -> HTMLResponse:
    """Image-similarity search.

    Always operates on a single film's index (per-film or flat
    ``from_config`` back-compat).  Aggregate image search is deferred
    to a later plan.
    """
    cfg = get_config()
    ctx = FilmContext.for_film(cfg, slug) if slug is not None else FilmContext.from_config(cfg)
    index = search_service.load_index(
        ctx,
        mapping_filename=cfg.embeddings.mapping_filename,
        embeddings_filename=cfg.embeddings.filename,
    )
    if not index.ok:
        return _no_index_response(request)

    data = await file.read()
    try:
        suffix = search_service.validate_upload(file.filename, file.content_type, data)
    except search_service.UploadRejected as exc:
        logger.info("Image-search upload rejected: %s", exc)
        return templates.TemplateResponse(
            request,
            "partials/search_results.html",
            make_ctx(
                request,
                current_slug=slug,
                results=[],
                no_index=False,
                upload_error=str(exc),
                films_by_id=search_service.films_by_id_lookup(cfg),
            ),
        )

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)

    try:
        loop = asyncio.get_event_loop()
        results_df = await loop.run_in_executor(
            None, search_service.search_image, index, tmp_path, top_k
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    meta_by_scene, fps = _kf_meta(ctx)
    results = _enrich_with_film_metadata(
        cfg,
        search_service.results_to_dicts(results_df, ctx.data_dir, meta_by_scene, fps),
        per_film_slug=slug,
    )
    return templates.TemplateResponse(
        request,
        "partials/search_results.html",
        make_ctx(
            request,
            current_slug=slug,
            results=results,
            no_index=False,
            films_by_id=search_service.films_by_id_lookup(cfg),
        ),
    )
