"""Search tab routes — text + image semantic search via CLIP.

Thin HTTP layer: parse, executor-offload, render. Logic in
:mod:`cinemateca.search`; render helpers in :mod:`api.services._search_render`.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Query, Request, UploadFile
from fastapi.responses import HTMLResponse

from api.deps import film_slug_query, get_config, make_ctx, optional_film_context
from api.schemas import SearchParams
from api.services import search as search_service
from api.services._field_errors import upload_error_response
from api.services._search_query import dispatch_search as _dispatch_search
from api.services._search_render import build_search_context
from api.services._search_render import enriched_per_film as _enriched_per_film
from api.services._search_render import no_index_response as _no_index_response
from api.services._search_render import render_results as _render_results
from api.templates import templates
from cinemateca.library import FilmContext

logger = logging.getLogger(__name__)
router = APIRouter()


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
    params: SearchParams = Depends(SearchParams),
    tags: list[str] = Query(default=[]),
    slug: str | None = Depends(film_slug_query),
    ctx: FilmContext | None = Depends(optional_film_context),
    offset: int = Query(default=0, ge=0),  # A7 paging: zero-based result offset
) -> HTMLResponse:
    """Semantic search across one (``?film=<slug>``) or all films.

    Orchestration (validate → dispatch → accessible inline-error swap) lives
    in ``_search_query.dispatch_search`` so this stays HTTP-shape only.
    """
    return await _dispatch_search(
        request, params=params, tags=list(tags), slug=slug, ctx=ctx, offset=offset
    )


@router.post("/api/search/image", response_class=HTMLResponse)
async def api_search_image(
    request: Request,
    file: UploadFile = File(...),
    top_k: int = Query(default=8, ge=1, le=200),  # bounded like text search (SearchParams)
    slug: str | None = Depends(film_slug_query),
    ctx: FilmContext | None = Depends(optional_film_context),
) -> HTMLResponse:
    """Image-similarity search. Upload validated first (→400 before index check).

    On rejection (U1) the response keeps its honest 400 status (pinned by
    ``test_image_upload_rejection_is_4xx``) but its body is the accessible
    inline field-error fragment targeting ``#image-upload-error`` via an OOB
    swap. The ``htmx:beforeSwap`` shim in mojica.js permits that fragment to
    apply despite the 4xx (HTMX suppresses body swaps on error codes by
    default), so the user sees a field-level message instead of the page-level
    error envelope.
    """
    # Validate before loading the index: a bad file is a 400 regardless of state.
    data = await file.read(search_service.MAX_UPLOAD_BYTES + 1)
    if len(data) > search_service.MAX_UPLOAD_BYTES:
        logger.info(
            "Image-search upload rejected: file too large (%d bytes > %d limit)",
            len(data),
            search_service.MAX_UPLOAD_BYTES,
        )
        return upload_error_response(request, "upload_too_large")
    try:
        suffix = search_service.validate_upload(file.filename, file.content_type, data)
    except search_service.UploadRejected as exc:
        logger.info("Image-search upload rejected: %s", exc)
        return upload_error_response(request, "upload_unsupported")

    cfg = get_config()
    loop = asyncio.get_running_loop()
    tmp_path: Path | None = None
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    try:
        if ctx is None:
            # Aggregate: search across all registered films (each has its own
            # current index).  flat_film_context() used the legacy
            # data/embeddings/ index whose paths pointed to the pre-library
            # data/frames/ tree — now empty — producing black cards.
            cards = await loop.run_in_executor(
                None, search_service.aggregate_image_search, cfg, tmp_path, top_k
            )
            results = search_service.enrich_hits_with_film_metadata(cfg, cards)
        else:
            index = search_service.load_index(
                ctx,
                mapping_filename=cfg.embeddings.mapping_filename,
                embeddings_filename=cfg.embeddings.filename,
                cfg=cfg,
            )
            if not index.ok:
                return _no_index_response(request)
            results_df = await loop.run_in_executor(
                None, search_service.search_image, index, tmp_path, top_k
            )
            results = _enriched_per_film(cfg, ctx, results_df, slug)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
    return _render_results(request, slug=slug, cfg=cfg, results=results)
