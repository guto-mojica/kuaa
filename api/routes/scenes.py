"""Scenes tab routes — Cenas (Mojica redesign) browsing endpoints.

Thin HTTP layer: request parsing + template rendering only. The Cenas
context is built by :func:`api.services.scenes_service.build_cenas_context`
which loads per-film metadata, runs the ``tipo_of`` classifier, and
groups scenes by film for the new ``.c-cp`` markup.

T9: Routes accept an optional ``?film=<slug>`` query parameter.
``slug=None`` → aggregate view across all registered films;
``slug=<value>`` → narrowed to a single film's group (unknown slug →
``ValueError`` from ``FilmContext.for_film``, legacy contract).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

from api.deps import film_slug_query, get_config, make_ctx
from api.services.scenes_service import (
    build_cenas_context,
    build_inspector_context,
)
from api.templates import templates

logger = logging.getLogger(__name__)
router = APIRouter()


def _parse_selected_scene_id(request: Request) -> int | None:
    """Return ``?scene=<int>`` from the request, or ``None`` if absent/invalid.

    Mirrors the search-page convention (``/search?scene=<id>&film=<slug>``)
    so the Cenas tab can deep-link a selected card via the URL bar. An
    invalid integer is treated as no selection (silently ignored) rather
    than 400ing — robustness over strictness for a UX-only param.
    """
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
) -> HTMLResponse:
    """Render the Cenas (Scenes) tab partial.

    ``slug=None`` → library-wide grouped grid (one ``.group`` heading
    per film). ``slug=<value>`` → narrowed to a single film's group;
    unknown slugs surface a ``ValueError`` from ``FilmContext.for_film``
    (the legacy contract pinned by ``test_tab_scenes_unknown_slug_raises``).
    The redesigned grid keeps the same visual scaffolding either way —
    when only one film matches the user still sees the per-film header
    row above its scenecards.
    """
    cfg = get_config()
    selected_scene_id = _parse_selected_scene_id(request)
    context = build_cenas_context(cfg, selected_scene_id=selected_scene_id, slug=slug)
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
    """Return the filtered Cenas grid fragment for HTMX swaps.

    The toolrow's ``.find`` input fires ``GET /api/scenes?q=<query>``
    on keyup; legacy tag-filter callers still pass ``tags[]``. The
    response is the new ``scenes_grid.html`` partial — film headings +
    scenecards in the Mojica markup. ``?film=<slug>`` narrows the
    result to a single film's group; unknown slugs surface a
    ``ValueError`` (legacy contract).
    """
    cfg = get_config()
    selected_scene_id = _parse_selected_scene_id(request)
    ctx = build_cenas_context(
        cfg,
        tags=tags,
        keyword=q,
        selected_scene_id=selected_scene_id,
        slug=slug,
    )
    # The grid partial only needs ``groups_by_film`` +
    # ``selected_scene_id``; slim the dict so callers don't bloat
    # the HTMX response headers.
    grid_ctx = {
        "groups_by_film": ctx["groups_by_film"],
        "selected_scene_id": ctx["selected_scene_id"],
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
