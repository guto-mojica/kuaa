"""Scenes tab routes — catalogue browsing with tag and keyword filters.

Thin HTTP layer: request parsing + template rendering only. All JSON
loading, scene-id normalization, keyframe-URL math and card
construction live in ``api/services/catalog.py`` (Phase 3a). Path
resolution flows through ``FilmContext``.

T9: Routes accept an optional ``?film=<slug>`` query parameter.
``slug=None`` → aggregate view across all registered films;
``slug=<value>`` → per-film view via ``FilmContext.for_film``.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

from api.deps import film_slug_query, get_config, make_ctx
from api.services.catalog import (
    build_scenes_context,
    build_scenes_context_aggregate,
    build_scenes_grid,
    build_scenes_grid_aggregate,
)
from api.services.film_context import FilmContext
from api.services.scenes_service import build_inspector_context
from api.templates import templates

logger = logging.getLogger(__name__)
router = APIRouter()


def _film_ctx(cfg, slug: str | None) -> FilmContext:
    """Return a ``FilmContext`` for *slug* (per-film) or flat config (no slug)."""
    if slug is not None:
        return FilmContext.for_film(cfg, slug)
    return FilmContext.from_config(cfg)


@router.get("/tab/scenes", response_class=HTMLResponse)
async def tab_scenes(
    request: Request,
    slug: str | None = Depends(film_slug_query),
) -> HTMLResponse:
    cfg = get_config()
    if slug is None:
        context = build_scenes_context_aggregate(cfg)
    else:
        context = build_scenes_context(FilmContext.for_film(cfg, slug))
    return templates.TemplateResponse(
        request,
        "partials/scenes.html",
        make_ctx(request, current_slug=slug, **context),
    )


@router.get("/api/scenes", response_class=HTMLResponse)
async def api_scenes(
    request: Request,
    tags: list[str] = Query(default=[]),
    q: str = "",
    slug: str | None = Depends(film_slug_query),
) -> HTMLResponse:
    cfg = get_config()
    if slug is None:
        grid = build_scenes_grid_aggregate(cfg, tags, q)
    else:
        grid = build_scenes_grid(FilmContext.for_film(cfg, slug), tags, q)
    return templates.TemplateResponse(
        request,
        "partials/scenes_grid.html",
        make_ctx(request, current_slug=slug, **grid),
    )


@router.get("/api/scenes/{scene_id}/inspector", response_class=HTMLResponse)
async def api_scene_inspector(
    request: Request,
    scene_id: int,
    tab: str = Query(default="activity"),
    slug: str | None = Depends(film_slug_query),
) -> HTMLResponse:
    """Render the right-pane Buscar inspector for ``scene_id`` (HTMX swap target).

    Task 12 of the Mojica redesign: every ``.b-card`` in the search
    results sets ``hx-get="/api/scenes/<id>/inspector?film=<slug>"`` and
    targets ``#right-pane``. This endpoint resolves the (slug, scene_id)
    pair through :func:`build_inspector_context`, renders
    ``partials/search_inspector.html``, and returns the HTMX fragment.

    Unresolvable pairs (unknown slug, unknown scene_id, missing
    per-film metadata) return a 404 — the result card stays selected on
    the page and the inspector simply does not swap.
    """
    cfg = get_config()
    ctx = build_inspector_context(cfg, scene_id=scene_id, slug=slug, inspector_tab=tab)
    if ctx is None:
        raise HTTPException(status_code=404, detail="scene not found")
    return templates.TemplateResponse(
        request,
        "partials/search_inspector.html",
        make_ctx(request, current_slug=slug, **ctx),
    )
