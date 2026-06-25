"""Search render helpers extracted from ``api/routes/search.py`` (A2 / Task 5).

These functions handle result enrichment and template assembly. The route
keeps only the FastAPI handlers and param parsing; all rendering logic lives
here.

``build_search_context`` lives here (moved from the route — G1 LOC fix) so
``api/server.py`` can continue to call it via ``search.build_search_context``
(the route re-imports it into its own namespace).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import Request
from fastapi.responses import HTMLResponse

from api.contexts import SearchContext
from api.deps import get_config, make_ctx, resolve_film_context
from api.services import search as search_service
from api.services.catalog import derive_fps, load_json
from api.templates import templates
from kuaa.library import FilmContext

logger = logging.getLogger(__name__)


def build_search_context(slug: str | None = None) -> SearchContext:
    """Build the Buscar tab template context for one film or the whole library.

    Re-exported from ``api.routes.search`` so ``api/server.py`` can call
    ``search.build_search_context(slug)`` without change.
    """
    cfg = get_config()
    if slug is not None:
        return search_service.build_search_context(resolve_film_context(cfg, slug, None), cfg)
    return search_service.build_search_context_aggregate(cfg)


def kf_meta(ctx: FilmContext) -> tuple[dict, float]:
    """Return ``(scene_id→entry dict, fps)`` from keyframes_metadata.json."""
    raw = load_json(ctx.metadata_dir / "keyframes_metadata.json")
    kf: list[Any] = raw if isinstance(raw, list) else []
    return {e["scene_id"]: e for e in kf if "scene_id" in e}, derive_fps(kf)


def render_search(request: Request, **ctx) -> HTMLResponse:
    """Return the ``partials/search_results.html`` template response."""
    return templates.TemplateResponse(
        request, "partials/search_results.html", make_ctx(request, **ctx)
    )


def no_index_response(request: Request) -> HTMLResponse:
    """Return the no-index empty-state response."""
    return render_search(request, results=[], no_index=True, films_by_id={})


def render_results(req: Request, *, slug, cfg, results, **extra) -> HTMLResponse:
    """Enrich and render the results partial.

    Sets ``HX-Replace-Url`` to a navigable page URL (``/search?q=…``) when a
    text query is present so the browser URL bar stays bookmarkable without
    pushing the internal ``/api/search?…`` API URL into history.  Image-search
    calls omit ``query`` in ``extra``, so they leave the URL bar untouched.
    """
    from urllib.parse import urlencode

    fbi = search_service.films_by_id_lookup(cfg)
    resp = render_search(
        req, current_slug=slug, results=results, no_index=False, films_by_id=fbi, **extra
    )
    q = (extra.get("query") or "").strip()
    if q:
        params: dict[str, str] = {}
        if slug:
            params["film"] = slug
        params["q"] = q
        resp.headers["HX-Replace-Url"] = "/search?" + urlencode(params)
    return resp


def enriched_per_film(cfg, ctx: FilmContext, results_df, slug: str | None) -> list[dict]:
    """Build per-film enriched result dicts from a raw results dataframe."""
    mbs, fps = kf_meta(ctx)
    dicts = search_service.results_to_dicts(results_df, ctx.data_dir, mbs, fps)
    return search_service.enrich_hits_with_film_metadata(cfg, dicts, per_film_slug=slug)


async def run_text_search(
    request: Request,
    *,
    q: str,
    slug: str | None,
    ctx: FilmContext | None,
    cfg: Any,
    tags: list[str],
    top_k: int,
    retriever: str,
    sem_w: float,
    bm25_w: float,
    rrf_k: int,
    reranker_enabled: bool | None,
    offset: int,
) -> HTMLResponse:
    """Text-search dispatch body, extracted from ``api_search`` (G1 LOC fix).

    Handles per-film and aggregate paths, reranking, and paging; returns
    the rendered results partial.
    """
    min_sim = float(getattr(cfg.embeddings, "min_similarity", 0.0) or 0.0)
    # Widen the first-stage candidate pool (1.1). Retrieving only ``top_k`` hits
    # (a) starves the cross-encoder — ``rerank`` scores at most ``top_k_in`` of
    # the hits it is handed, so without depth it can only reorder the page the
    # first stage already surfaced rather than promote a deeper candidate — and
    # (b) breaks ``offset`` paging, because the final ``[offset : offset+top_k]``
    # slice would index past a ``top_k``-length list and return nothing. Fetch
    # enough for both: the paging window plus, when reranking is on, the
    # reranker's input window. The reorder + page slice still happen below.
    rr_enabled, _rr_model, rr_top_k_in = search_service._reranker_settings(cfg, reranker_enabled)
    first_stage_k = top_k + offset
    if rr_enabled:
        first_stage_k = max(first_stage_k, rr_top_k_in)
    logger.info(
        "api_search q=%r slug=%s retriever=%s top_k=%d first_stage_k=%d min_sim=%.3f "
        "sw=%.3f bw=%.3f tags=%s reranker_enabled=%s offset=%d",
        q,
        slug or "(agg)",
        retriever,
        top_k,
        first_stage_k,
        min_sim,
        sem_w,
        bm25_w,
        list(tags) or None,
        reranker_enabled,
        offset,
    )
    args = (cfg, ctx, q, tags, first_stage_k, min_sim, retriever, sem_w, bm25_w, rrf_k)
    payload, no_index = await asyncio.get_running_loop().run_in_executor(
        None, lambda: search_service.dispatch_text_search(*args)
    )
    if no_index and (slug is not None or not payload):
        return no_index_response(request)
    if slug is None:
        agg = search_service.aggregate_hits_to_template_dicts(cfg, payload) if payload else []
        cards = search_service.enrich_hits_with_film_metadata(cfg, agg) if agg else []
    else:
        if ctx is None:
            return no_index_response(request)
        cards = enriched_per_film(cfg, ctx, payload, slug)
    # C5: carry a typed SearchResult from enrichment through rerank to the
    # render boundary. ``cards_to_result`` is the single dict→typed lift;
    # ``rerank_search_result`` operates on that result (no dict round-trip);
    # ``result_to_cards`` projects it back for the HTML template, which still
    # reads display-only fields (img_url / similarity / pin_count) off the
    # card dicts rather than the core ``Hit``.
    result, originals = search_service.cards_to_result(cards, query=q, mode=retriever)
    result = search_service.rerank_search_result(result, cfg=cfg, enabled=reranker_enabled)
    results = search_service.result_to_cards(result, originals)
    results = results[offset : offset + top_k]
    return render_results(
        request, slug=slug, cfg=cfg, results=results, query=q, highlighted_tags=set(tags)
    )
