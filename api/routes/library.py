"""Library sidebar route — registry filter.

Shows the registry-backed film list (films.json) filtered by name.
Per-film scene counts and processed state are REAL (read from
``<library_dir>/<slug>/metadata/keyframes_metadata.json``).
Full per-film selection affordances land in T9/T10.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from api.deps import get_config, make_ctx
from api.templates import templates

router = APIRouter()


def _library_ctx(request: Request, q: str = "") -> dict:
    """Build the sidebar context: global state + filtered registry film list."""
    from cinemateca.library import library_state, scan_library

    cfg = get_config()
    # TODO(T5): switch to cfg.paths.library_dir once the config knob lands.
    library_dir = Path(cfg.paths.data_dir)
    raw_dir = Path(cfg.paths.raw_dir)
    metadata_dir = Path(cfg.paths.metadata_dir)

    films = scan_library(library_dir)
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
