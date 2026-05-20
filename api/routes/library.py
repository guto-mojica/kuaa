"""Library sidebar route — registry filter.

Shows the registry-backed film list (films.json) filtered by name.
Per-film scene counts and processed state are REAL (read from
``<library_dir>/<slug>/metadata/keyframes_metadata.json``).

T9: ``/api/library/filter`` accepts an optional ``?film=<slug>`` query
parameter (wired for completeness; the filter route always returns the
full library tree, not a per-film subtree).
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse

from api.deps import film_slug_query, get_config, make_ctx
from api.templates import templates

router = APIRouter()


def _library_ctx(request: Request, q: str = "") -> dict:
    """Build the sidebar context: global state + filtered registry film list."""
    from cinemateca.library import library_state, scan_library

    cfg = get_config()
    library_dir = Path(cfg.paths.library_dir)

    films = scan_library(library_dir)
    if q.strip():
        needle = q.strip().lower()
        films = [f for f in films if needle in f.title.lower() or needle in f.slug.lower()]

    state = library_state(library_dir)
    return make_ctx(request, films=films, library_state=state)


@router.get("/api/library/filter", response_class=HTMLResponse)
async def api_library_filter(
    request: Request,
    q: str = "",
    slug: str | None = Depends(film_slug_query),
) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "partials/library_tree.html",
        _library_ctx(request, q),
    )
