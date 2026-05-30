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

from api.deps import film_slug_query, flat_film_context, get_config, make_ctx, optional_film_context
from api.schemas import SearchParams
from api.services import search as search_service
from api.services._search_render import api_search_audio as _api_search_audio
from api.services._search_render import api_search_fusion as _api_search_fusion
from api.services._search_render import build_search_context
from api.services._search_render import enriched_per_film as _enriched_per_film
from api.services._search_render import no_index_response as _no_index_response
from api.services._search_render import render_results as _render_results
from api.services._search_render import run_text_search as _run_text_search
from api.templates import templates
from cinemateca.errors import UserInputError
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
    """Semantic search across one (``?film=<slug>``) or all films."""
    q = params.q.strip()
    if len(q) < 2:
        return HTMLResponse("")
    cfg = get_config()
    if params.modality == "audio":
        return await _api_search_audio(request, q=q, top_k=params.top_k, slug=slug, cfg=cfg)
    if params.modality == "fusion":
        return await _api_search_fusion(
            request, q=q, top_k=params.top_k, w=params.w, slug=slug, cfg=cfg
        )
    retriever, sw, bw, rrf_k = search_service.resolve_retriever_args(
        cfg, params.retriever, params.sem_w, params.bm25_w
    )
    return await _run_text_search(
        request,
        q=q,
        slug=slug,
        ctx=ctx,
        cfg=cfg,
        tags=list(tags),
        top_k=params.top_k,
        retriever=retriever,
        sem_w=sw,
        bm25_w=bw,
        rrf_k=rrf_k,
        reranker_enabled=params.reranker_enabled,
        offset=offset,
    )


@router.post("/api/search/image", response_class=HTMLResponse)
async def api_search_image(
    request: Request,
    file: UploadFile = File(...),
    top_k: int = 8,
    slug: str | None = Depends(film_slug_query),
    ctx: FilmContext | None = Depends(optional_film_context),
) -> HTMLResponse:
    """Image-similarity search. Upload validated first (→400 before index check)."""
    # Validate before loading the index: a bad file is a 400 regardless of state.
    data = await file.read(search_service.MAX_UPLOAD_BYTES + 1)
    if len(data) > search_service.MAX_UPLOAD_BYTES:
        msg = f"file too large ({len(data)} bytes > {search_service.MAX_UPLOAD_BYTES} limit)"
        logger.info("Image-search upload rejected: %s", msg)
        raise UserInputError(msg)
    try:
        suffix = search_service.validate_upload(file.filename, file.content_type, data)
    except search_service.UploadRejected as exc:
        logger.info("Image-search upload rejected: %s", exc)
        raise UserInputError(str(exc)) from exc

    cfg = get_config()
    ctx = ctx if ctx is not None else flat_film_context()
    index = search_service.load_index(
        ctx,
        mapping_filename=cfg.embeddings.mapping_filename,
        embeddings_filename=cfg.embeddings.filename,
        cfg=cfg,
    )
    if not index.ok:
        return _no_index_response(request)
    tmp_path: Path | None = None
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    try:
        loop = asyncio.get_running_loop()
        results_df = await loop.run_in_executor(
            None, search_service.search_image, index, tmp_path, top_k
        )
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
    results = _enriched_per_film(cfg, ctx, results_df, slug)
    return _render_results(request, slug=slug, cfg=cfg, results=results)
