"""Library sidebar route — inventory filter only.

v0.3 is SINGLE-FILM: there is no per-film context to switch into, so
there is no film-"select" route. The sidebar shows the honest GLOBAL
artifact state plus the raw videos as a plain inventory; the only
interaction is filtering that inventory by name. (A real per-film
selection lands with the post-recovery multi-film epic; see
``api.services.film_context.FilmContext``.)
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from api.deps import get_config, make_ctx
from api.templates import templates

router = APIRouter()


def _library_ctx(request: Request, q: str = "") -> dict:
    """Build the sidebar context: honest global state + filtered inventory."""
    from cinemateca.library import library_state, scan_library

    cfg = get_config()
    raw_dir = Path(cfg.paths.raw_dir)
    metadata_dir = Path(cfg.paths.metadata_dir)

    films = scan_library(raw_dir=raw_dir, metadata_dir=metadata_dir)
    if q.strip():
        needle = q.strip().lower()
        films = [f for f in films if needle in f.title.lower() or needle in f.slug.lower()]

    state = library_state(
        raw_dir=raw_dir,
        metadata_dir=metadata_dir,
        embeddings_index_path=Path(cfg.paths.embeddings_dir) / cfg.embeddings.filename,
    )
    return make_ctx(request, films=films, library_state=state)


@router.get("/api/library/filter", response_class=HTMLResponse)
async def api_library_filter(request: Request, q: str = "") -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "partials/library_tree.html",
        _library_ctx(request, q),
    )
