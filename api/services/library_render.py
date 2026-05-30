"""Library context builders extracted from ``api/routes/library.py`` (A2 / Task 5).

These helpers build the template context for the library sidebar endpoints.
The route keeps only the FastAPI handler shapes.
"""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import HTMLResponse

from api.deps import get_config, make_ctx
from api.services.chrome_service import build_chrome_context
from api.templates import templates


def library_ctx(request: Request, q: str = "", current_slug: str | None = None) -> dict:
    """Build the legacy sidebar context: global state + filtered registry film list."""
    from pathlib import Path

    from cinemateca.library import library_state, scan_library

    cfg = get_config()
    library_dir = Path(cfg.paths.library_dir)

    films = scan_library(library_dir)
    if q.strip():
        needle = q.strip().lower()
        films = [f for f in films if needle in f.title.lower() or needle in f.slug.lower()]

    state = library_state(library_dir)
    return make_ctx(request, films=films, library_state=state, current_slug=current_slug)


def chrome_filter_ctx(request: Request, q: str = "", current_slug: str | None = None) -> dict:
    """Build the Mojica LeftPane context for the /api/library/tree endpoint.

    Reuses :func:`build_chrome_context` so the filtered fragment carries
    the same collections / job-slug / runtime context as the initial
    server-side include. The string filter is applied AFTER the chrome
    bag is built so the unfiltered ``library_state`` and runtime stats
    (rendered in the footer of the parent ``_left_pane.html``) are
    unchanged — only the films list inside ``.scroll`` is narrowed.
    """
    cfg = get_config()
    chrome = build_chrome_context(cfg, current_slug=current_slug)
    if q.strip():
        needle = q.strip().lower()
        chrome["films"] = [
            f for f in chrome["films"] if needle in f.title.lower() or needle in f.slug.lower()
        ]
    return make_ctx(request, **chrome)


def tree_response(request: Request) -> HTMLResponse:
    """Return the legacy library_tree partial with an unfiltered context."""
    return templates.TemplateResponse(
        request,
        "partials/library_tree.html",
        library_ctx(request),
    )
