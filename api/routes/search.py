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

from api.deps import film_slug_query, flat_film_context, get_config, make_ctx, optional_film_context, resolve_film_context  # noqa: E501
from api.schemas import SearchParams
from cinemateca.errors import UserInputError
from api.services import search as search_service
from api.services._search_render import (
    api_search_audio as _api_search_audio,
    api_search_fusion as _api_search_fusion,
    enriched_per_film as _enriched_per_film,
    no_index_response as _no_index_response,
    render_results as _render_results,
)
from api.templates import templates

logger = logging.getLogger(__name__)
router = APIRouter()


def build_search_context(slug: str | None = None) -> dict:
    """Re-exported for ``api/server.py``'s tab-context map."""
    cfg = get_config()
    if slug is not None:
        return search_service.build_search_context(resolve_film_context(cfg, slug, None), cfg)
    return search_service.build_search_context_aggregate(cfg)


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
    min_sim = float(getattr(cfg.embeddings, "min_similarity", 0.0) or 0.0)
    retriever, sw, bw, rrf_k = search_service.resolve_retriever_args(
        cfg, params.retriever, params.sem_w, params.bm25_w
    )
    logger.info(
        f"api_search q={q!r} slug={slug or '(agg)'} retriever={retriever} top_k={params.top_k} "
        f"min_sim={min_sim:.3f} sw={sw:.3f} bw={bw:.3f} tags={list(tags) or None} "
        f"reranker_enabled={params.reranker_enabled}"
    )
    args = (cfg, ctx, q, tags, params.top_k, min_sim, retriever, sw, bw, rrf_k)
    payload, no_index = await asyncio.get_running_loop().run_in_executor(
        None, lambda: search_service.dispatch_text_search(*args)
    )
    if no_index and (slug is not None or not payload):
        return _no_index_response(request)
    if slug is None:
        agg = search_service.aggregate_hits_to_template_dicts(cfg, payload) if payload else []
        results = search_service.enrich_hits_with_film_metadata(cfg, agg) if agg else []
    else:
        results = _enriched_per_film(cfg, ctx, payload, slug)
    results = search_service.rerank_template_results(
        results,
        cfg=cfg,
        query=q,
        mode=retriever,
        enabled=params.reranker_enabled,
    )
    return _render_results(
        request, slug=slug, cfg=cfg, results=results, query=q,
        highlighted_tags=set(tags),
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
