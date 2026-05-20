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
def build_search_context() -> dict:
    """Build the search-tab partial context (delegates to the service)."""
    ctx = FilmContext.from_config(get_config())
    return search_service.build_search_context(ctx)


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
        make_ctx(request, results=[], no_index=True),
    )


@router.get("/tab/search", response_class=HTMLResponse)
async def tab_search(
    request: Request,
    slug: str | None = Depends(film_slug_query),
) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "partials/search.html",
        make_ctx(request, current_slug=slug, **build_search_context()),
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
                    cfg, query=q, modality="text", top_k=top_k
                ),
            )
        except NotImplementedError:
            return _no_index_response(request)
        if not hits:
            return _no_index_response(request)
        # Convert aggregate hit dicts → template-compatible result dicts.
        # aggregate_search returns plain dicts (not DataFrames), so
        # results_to_dicts is not applicable; build the img_url here.
        library_dir = Path(cfg.paths.library_dir).resolve()
        results = [
            {
                **h,
                "img_url": search_service.keyframe_url(h["keyframe_path"], library_dir),
            }
            for h in hits
        ]
        return templates.TemplateResponse(
            request,
            "partials/search_results.html",
            make_ctx(request, current_slug=slug, results=results, no_index=False),
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
        None, search_service.search_text, index, q, tags, tag_index, top_k
    )

    meta_by_scene, fps = _kf_meta(ctx)
    return templates.TemplateResponse(
        request,
        "partials/search_results.html",
        make_ctx(
            request,
            current_slug=slug,
            results=search_service.results_to_dicts(
                results_df, ctx.data_dir, meta_by_scene, fps
            ),
            no_index=False,
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
        suffix = search_service.validate_upload(
            file.filename, file.content_type, data
        )
    except search_service.UploadRejected as exc:
        logger.info("Image-search upload rejected: %s", exc)
        return templates.TemplateResponse(
            request,
            "partials/search_results.html",
            make_ctx(request, current_slug=slug, results=[], no_index=False, upload_error=str(exc)),
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
    return templates.TemplateResponse(
        request,
        "partials/search_results.html",
        make_ctx(
            request,
            current_slug=slug,
            results=search_service.results_to_dicts(
                results_df, ctx.data_dir, meta_by_scene, fps
            ),
            no_index=False,
        ),
    )
