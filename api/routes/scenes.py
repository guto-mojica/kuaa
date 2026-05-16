"""Scenes tab routes — catalogue browsing with tag and keyword filters.

Thin HTTP layer: request parsing + template rendering only. All JSON
loading, scene-id normalization, keyframe-URL math and card
construction live in ``api/services/catalog.py`` (Phase 3a). Path
resolution flows through ``FilmContext``.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse

from api.deps import get_config, make_ctx
from api.services.catalog import build_scenes_context, build_scenes_grid
from api.services.film_context import FilmContext
from api.templates import templates

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/tab/scenes", response_class=HTMLResponse)
async def tab_scenes(request: Request) -> HTMLResponse:
    ctx = FilmContext.from_config(get_config())
    return templates.TemplateResponse(
        request,
        "partials/scenes.html",
        make_ctx(request, **build_scenes_context(ctx)),
    )


@router.get("/api/scenes", response_class=HTMLResponse)
async def api_scenes(
    request: Request,
    tags: list[str] = Query(default=[]),
    q: str = "",
) -> HTMLResponse:
    ctx = FilmContext.from_config(get_config())
    return templates.TemplateResponse(
        request,
        "partials/scenes_grid.html",
        make_ctx(request, **build_scenes_grid(ctx, tags, q)),
    )
