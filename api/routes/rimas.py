"""Rimas Visuais (cross-film visual rhymes) tab routes — Phase-5.

Thin HTTP layer: param parsing + template dispatch only. Context builders
live in :mod:`api.services.rhymes_service` (A2 Task 5).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse

from api.contexts import RimasContext
from api.deps import get_config, make_ctx
from api.services.rhymes_service import build_rimas_context
from api.templates import templates

logger = logging.getLogger(__name__)
router = APIRouter()


def _rimas_ctx(cfg, anchor, echo, lambda_, k_candidates) -> RimasContext:
    """Assemble rimas context from route params — single call for all three handlers."""
    return build_rimas_context(
        cfg,
        anchor=anchor,
        echo=echo,
        lambda_diversity=lambda_,
        k_candidates=k_candidates,
    )


@router.get("/tab/rimas", response_class=HTMLResponse)
async def tab_rimas(
    request: Request,
    anchor: str | None = None,
    echo: str | None = None,
    lambda_: float | None = Query(default=None, alias="lambda"),
    k_candidates: int | None = None,
) -> HTMLResponse:
    """Render the Rimas Visuais tab partial."""
    ctx = _rimas_ctx(get_config(), anchor, echo, lambda_, k_candidates)
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
    """Return the echo-grid fragment for HTMX anchor swaps."""
    ctx = _rimas_ctx(get_config(), anchor, echo, lambda_, k_candidates)
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
    """Return the right-pane inspector fragment."""
    ctx = _rimas_ctx(get_config(), anchor, echo, lambda_, k_candidates)
    return templates.TemplateResponse(
        request,
        "partials/rimas_inspector.html",
        make_ctx(request, **ctx),
    )
