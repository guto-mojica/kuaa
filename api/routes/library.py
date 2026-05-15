"""Library sidebar routes — filter and film selection."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse

from api.deps import get_config, make_ctx
from api.templates import templates

router = APIRouter()


@router.get("/api/library/filter", response_class=HTMLResponse)
async def api_library_filter(request: Request, q: str = "") -> HTMLResponse:
    from cinemateca.library import scan_library

    cfg = get_config()
    films = scan_library(
        raw_dir=Path(cfg.paths.raw_dir),
        metadata_dir=Path(cfg.paths.metadata_dir),
    )
    if q.strip():
        needle = q.strip().lower()
        films = [f for f in films if needle in f.title.lower() or needle in f.slug.lower()]

    return templates.TemplateResponse(
        request,
        "partials/library_tree.html",
        make_ctx(request, films=films, selected_slug=None),
    )


@router.get("/api/library/{slug}/select", response_class=HTMLResponse)
async def api_library_select(slug: str, request: Request) -> HTMLResponse:
    """Selecting a film in the sidebar loads the Search tab."""
    return templates.TemplateResponse(
        request,
        "partials/search.html",
        make_ctx(request),
    )
