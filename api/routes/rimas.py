"""Rimas Visuais (cross-film visual rhymes) tab routes — Phase-5.

Thin HTTP layer: param parsing + template dispatch only. The context
itself is built by :func:`api.services.rhymes_service.build_rimas_context`,
which walks the library, resolves the anchor scene, and runs the
cross-film cosine kNN.

Two endpoints:

  * ``GET /tab/rimas`` — full-tab partial swap (the HTMX hx-target the
    chrome's tab-bar fires when the user clicks the Rimas tab chip).
    Renders ``partials/rimas.html`` with the complete page context
    (anchor + echoes + knobs).

  * ``GET /api/rimas/echoes`` — echo-grid fragment only. Used by
    Task 22's anchor-switch interactions (clicking a film in the
    sidebar or a candidate echo to "promote" it to the anchor swaps
    only the echo grid, not the whole tab). Renders
    ``partials/rimas_echoes.html`` against the same context shape.

The ``?anchor=<slug>/<scene_id>`` query param controls which scene the
service treats as the anchor; the service handles parsing + falling
back to a default anchor + the empty-state branch (no params crash, no
500s on unresolvable anchors).

The Phase-1 placeholder route in ``api/server.py`` (``page_rimas``)
remains: the full-page ``GET /rimas`` flows through ``render_page`` and
into ``_TAB_CONTEXT_BUILDERS['rimas']``. Task 22 swaps that placeholder
builder for :func:`api.services.rhymes_service.build_rimas_context`
once the page template lands; for Task 21 the tab-fragment endpoints
defined here are enough to exercise the service end-to-end.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse

from api.deps import get_config, make_ctx
from api.services.rhymes_service import build_rimas_context
from api.templates import templates

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/tab/rimas", response_class=HTMLResponse)
async def tab_rimas(
    request: Request,
    anchor: str | None = None,
    echo: str | None = None,
    lambda_: float | None = Query(default=None, alias="lambda"),
    k_candidates: int | None = None,
) -> HTMLResponse:
    """Render the Rimas Visuais tab partial.

    The service treats an absent / malformed ``anchor`` query param as
    "use the default anchor" — the first processed film, scene 1. When
    no film qualifies the partial renders an empty state. ``?echo=``
    deep-shares a specific rhyme card and pre-populates the inspector.

    ``?lambda=`` overrides ``cfg.retrieval.rhymes.diversity`` (the MMR
    relevance↔diversity trade-off; clamped to ``[0, 1]`` service-side);
    ``?k_candidates=`` overrides the pre-MMR pool size (default 30).
    Both default to ``None`` so the service falls back to config / hard
    defaults.
    """
    cfg = get_config()
    ctx = build_rimas_context(
        cfg,
        anchor=anchor,
        echo=echo,
        lambda_diversity=lambda_,
        k_candidates=k_candidates,
    )
    return templates.TemplateResponse(
        request,
        "partials/rimas.html",
        make_ctx(request, active_tab="rimas", **ctx),
    )


@router.get("/api/rimas/echoes", response_class=HTMLResponse)
async def api_rimas_echoes(
    request: Request,
    anchor: str | None = None,
    echo: str | None = None,
    lambda_: float | None = Query(default=None, alias="lambda"),
    k_candidates: int | None = None,
) -> HTMLResponse:
    """Return the echo-grid fragment for HTMX anchor swaps.

    The fragment is consumed by anchor-switch interactions: clicking a
    film in the sidebar or promoting a candidate echo to anchor fires
    this endpoint with the new ``?anchor=`` value and swaps the grid in
    place without reloading the whole tab.

    Shares the full context shape with :func:`tab_rimas` so the
    fragment can read every key the page template can (the echoes
    partial intentionally reads only ``echoes`` + the knob trio +
    ``selected_echo_id`` for the ``.sel`` highlight class; the rest is
    no-op).

    ``?lambda=`` and ``?k_candidates=`` mirror :func:`tab_rimas`: the
    diversity slider and the pre-MMR pool size, both threaded straight
    into the service.
    """
    cfg = get_config()
    ctx = build_rimas_context(
        cfg,
        anchor=anchor,
        echo=echo,
        lambda_diversity=lambda_,
        k_candidates=k_candidates,
    )
    return templates.TemplateResponse(
        request,
        "partials/rimas_echoes.html",
        make_ctx(request, **ctx),
    )


@router.get("/api/rimas/inspector", response_class=HTMLResponse)
async def api_rimas_inspector(
    request: Request,
    anchor: str | None = None,
    echo: str | None = None,
    lambda_: float | None = Query(default=None, alias="lambda"),
    k_candidates: int | None = None,
) -> HTMLResponse:
    """Return the right-pane inspector fragment.

    Fired by the .r-echo cards on click (hx-target="#right-pane",
    hx-swap="innerHTML"). The fragment renders the .r-pair comparison +
    similarity card + shared tags + actions for the selected echo, or
    an "anchor-only" empty branch when ``?echo=`` is omitted or
    unresolvable.

    Same 200-only contract as the other rimas endpoints: any
    unresolvable input collapses to an empty-but-rendered inspector,
    never a 500.

    ``?lambda=`` and ``?k_candidates=`` mirror :func:`tab_rimas` so the
    inspector stays in sync when the user scrubs the diversity slider
    or the pool size on the echo grid.
    """
    cfg = get_config()
    ctx = build_rimas_context(
        cfg,
        anchor=anchor,
        echo=echo,
        lambda_diversity=lambda_,
        k_candidates=k_candidates,
    )
    return templates.TemplateResponse(
        request,
        "partials/rimas_inspector.html",
        make_ctx(request, **ctx),
    )
