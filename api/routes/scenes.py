"""Scenes tab routes — Cenas (Mojica redesign) browsing endpoints.

Thin HTTP layer: request parsing + template rendering only. Context
builders live in :mod:`api.services.scenes` (A2 Task 5).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

from api.deps import film_slug_query, get_config, make_ctx
from api.schemas import Pagination
from api.services.scenes import (
    build_cenas_context,
    build_inspector_context,
    resolve_inspector_template,
)
from api.templates import templates

logger = logging.getLogger(__name__)
router = APIRouter()


def _parse_scene_id(request: Request) -> int | None:
    """Return ``?scene=<int>`` or ``None`` if absent/invalid (silently tolerant)."""
    raw = request.query_params.get("scene")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


@router.get("/tab/scenes", response_class=HTMLResponse)
async def tab_scenes(
    request: Request,
    slug: str | None = Depends(film_slug_query),
    group: str = "film",
    sort: str = "timecode",
    bucket: str | None = None,
) -> HTMLResponse:
    """Render the Cenas (Scenes) tab partial."""
    cfg = get_config()
    context = build_cenas_context(
        cfg,
        selected_scene_id=_parse_scene_id(request),
        slug=slug,
        group=group,
        sort=sort,
        bucket=bucket,
    )
    context["sentinel_sort"] = context["active_sort"]
    context["sentinel_group"] = context["active_group"]
    context["sentinel_bucket"] = context.get("active_bucket") or ""
    context["sentinel_q"] = ""
    context["sentinel_tags"] = []
    context["sentinel_slug"] = slug or ""
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
    group: str = "film",
    sort: str = "timecode",
    bucket: str | None = None,
    page: Pagination = Depends(Pagination),
    append: bool = Query(default=False),
) -> HTMLResponse:
    """Return the filtered Cenas grid fragment for HTMX swaps.

    ``append=true`` is set by the load-more sentinel: the grid partial
    suppresses group headers so they don't repeat when a film spans pages.
    """
    cfg = get_config()
    ctx = build_cenas_context(
        cfg,
        tags=tags,
        keyword=q,
        selected_scene_id=_parse_scene_id(request),
        slug=slug,
        group=group,
        sort=sort,
        bucket=bucket,
        limit=page.limit,
        offset=page.offset,
    )
    grid_ctx = {
        "groups_by_film": ctx["groups_by_film"],
        "selected_scene_id": ctx["selected_scene_id"],
        "total_scenes": ctx["total_scenes"],
        "has_more": ctx["has_more"],
        "has_prev": ctx["has_prev"],
        "current_offset": ctx["current_offset"],
        "next_offset": ctx["next_offset"],
        "prev_offset": ctx["prev_offset"],
        "current_limit": ctx["current_limit"],
        "is_append": append,
        # Sentinel self-containment: baked-in params that load-more must
        # preserve across pages (independent of the live Alpine store).
        "sentinel_sort": ctx["active_sort"],
        "sentinel_group": ctx["active_group"],
        "sentinel_bucket": ctx.get("active_bucket") or "",
        "sentinel_q": q or "",
        "sentinel_tags": list(tags),
        "sentinel_slug": slug or "",
    }
    return templates.TemplateResponse(
        request,
        "partials/scenes_grid.html",
        make_ctx(request, current_slug=slug, **grid_ctx),
    )


@router.get("/api/scenes/{scene_id}/inspector", response_class=HTMLResponse)
async def api_scene_inspector(
    request: Request,
    scene_id: int,
    tab: str = Query(default="activity"),
    kind: str = Query(default="buscar"),
    slug: str | None = Depends(film_slug_query),
) -> HTMLResponse:
    """Render the right-pane scene inspector (HTMX swap target)."""
    cfg = get_config()
    ctx = build_inspector_context(cfg, scene_id=scene_id, slug=slug, inspector_tab=tab)
    if ctx is None:
        raise HTTPException(status_code=404, detail="scene not found")
    template_name, inspector_kind = resolve_inspector_template(kind)
    ctx["inspector_kind"] = inspector_kind
    return templates.TemplateResponse(
        request,
        template_name,
        make_ctx(request, current_slug=slug, **ctx),
    )
