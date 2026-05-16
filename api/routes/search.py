"""Search tab routes — text and image semantic search via CLIP.

Thin HTTP layer: request parse + executor offload + template render.
All index loading (now mtime/size-cache-invalidated), shape validation,
upload guards and result conversion live in
``api/services/search.py`` (Phase 3c). Path resolution flows through
:class:`FilmContext` (consistent with scenes / annotate).
"""
from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Query, Request, UploadFile
from fastapi.responses import HTMLResponse

from api.deps import get_config, make_ctx
from api.services import search as search_service
from api.services.catalog import load_tag_index
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


def _no_index_response(request: Request) -> HTMLResponse:
    """Render the graceful no-index / corrupt-index results state."""
    return templates.TemplateResponse(
        request,
        "partials/search_results.html",
        make_ctx(request, results=[], no_index=True),
    )


@router.get("/tab/search", response_class=HTMLResponse)
async def tab_search(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "partials/search.html",
        make_ctx(request, **build_search_context()),
    )


@router.get("/api/search", response_class=HTMLResponse)
async def api_search(
    request: Request,
    q: str = "",
    tags: list[str] = Query(default=[]),
    top_k: int = 8,
) -> HTMLResponse:
    q = q.strip()
    if len(q) < 2:
        return HTMLResponse("")

    cfg = get_config()
    ctx = FilmContext.from_config(cfg)
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

    return templates.TemplateResponse(
        request,
        "partials/search_results.html",
        make_ctx(
            request,
            results=search_service.results_to_dicts(results_df, ctx.data_dir),
            no_index=False,
        ),
    )


@router.post("/api/search/image", response_class=HTMLResponse)
async def api_search_image(
    request: Request,
    file: UploadFile = File(...),
    top_k: int = 8,
) -> HTMLResponse:
    cfg = get_config()
    ctx = FilmContext.from_config(cfg)
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
            make_ctx(request, results=[], no_index=False, upload_error=str(exc)),
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

    return templates.TemplateResponse(
        request,
        "partials/search_results.html",
        make_ctx(
            request,
            results=search_service.results_to_dicts(results_df, ctx.data_dir),
            no_index=False,
        ),
    )
