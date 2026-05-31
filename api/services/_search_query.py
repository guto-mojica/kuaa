"""GET /api/search orchestration, extracted from the route (LOC budget).

Keeps ``api/routes/search.py`` HTTP-shape only. ``dispatch_search`` parses
the validated params, applies the U1 accessible inline query validation, and
fans out to the audio / fusion / text render paths in
:mod:`api.services._search_render`.
"""

from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.responses import HTMLResponse

from api.deps import get_config
from api.services import search as search_service
from api.services._field_errors import (
    prepend_oob,
    render_field_error_fragment,
    submit_triggered,
)
from api.services._search_render import api_search_audio, api_search_fusion, run_text_search
from cinemateca.library import FilmContext

_QUERY_ERROR_SLOT = "search-query-error"


async def dispatch_search(
    request: Request,
    *,
    params: Any,
    tags: list[str],
    slug: str | None,
    ctx: FilmContext | None,
    offset: int,
) -> HTMLResponse:
    """Validate → dispatch → render the search response.

    U1 accessible inline query validation: a query < 2 chars surfaces an OOB
    field-error into ``#search-query-error`` on an explicit SUBMIT
    (``query_empty`` / ``query_too_short``) but stays silent (empty slot) on a
    live keyup; a valid query dispatches to the audio / fusion / text path and
    prepends an OOB swap that CLEARS the slot.
    """
    q = params.q.strip()
    if len(q) < 2:
        if submit_triggered(request):
            key = "query_empty" if not q else "query_too_short"
        else:
            key = ""  # live keyup: clear the slot, show nothing
        return HTMLResponse(
            render_field_error_fragment(request, slot_id=_QUERY_ERROR_SLOT, message_key=key)
        )

    cfg = get_config()
    if params.modality == "audio":
        resp = await api_search_audio(request, q=q, top_k=params.top_k, slug=slug, cfg=cfg)
    elif params.modality == "fusion":
        resp = await api_search_fusion(
            request, q=q, top_k=params.top_k, w=params.w, slug=slug, cfg=cfg
        )
    else:
        retriever, sw, bw, rrf_k = search_service.resolve_retriever_args(
            cfg, params.retriever, params.sem_w, params.bm25_w
        )
        resp = await run_text_search(
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
    return prepend_oob(
        resp,
        render_field_error_fragment(request, slot_id=_QUERY_ERROR_SLOT, message_key=""),
    )
