"""Library sidebar routes — registry filter (legacy + Mojica chrome).

Two filter endpoints coexist during the Phase-1 / Phase-2 transition:

  * ``GET /api/library/filter`` — LEGACY. Returns ``library_tree.html``,
    the v0.3 sidebar tree (``.tree-node`` rows + add-film slot). Still
    wired to the legacy sidebar that ships inside ``.ch-main`` until
    Phase 2 deletes that block. Do NOT change its response shape — the
    legacy templates depend on it.

  * ``GET /api/library/tree`` — NEW (Task 8). Returns
    ``_left_pane_body.html``, the Mojica LeftPane content (films loop +
    collections + shared). Targeted by the new ``.ch-lp .filter`` input
    via ``hx-target=".ch-lp .scroll"``.

Both endpoints share the same per-film state source
(``cinemateca.library.scan_library``) and the same string-match filter
on ``title`` + ``slug``. The Mojica endpoint additionally returns
chrome-only context (collections, ``active_job_slugs``, …) so the
swapped fragment renders with the same vocabulary as the initial
include.

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

from api.deps import film_slug_query, get_config, get_library_dir, make_ctx
from api.services.chrome_service import build_chrome_context
from api.services.film_service import list_films
from api.templates import templates

router = APIRouter()


def _library_ctx(request: Request, q: str = "", current_slug: str | None = None) -> dict:
    """Build the legacy sidebar context: global state + filtered registry film list."""
    from cinemateca.library import library_state

    library_dir = get_library_dir()
    films = list_films(library_dir, q)
    state = library_state(library_dir)
    return make_ctx(request, films=films, library_state=state, current_slug=current_slug)


def _chrome_filter_ctx(request: Request, q: str = "", current_slug: str | None = None) -> dict:
    """Build the Mojica LeftPane context for the new /api/library/tree endpoint.

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


@router.get("/api/library/filter", response_class=HTMLResponse)
async def api_library_filter(
    request: Request,
    q: str = "",
    slug: str | None = Depends(film_slug_query),
) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "partials/library_tree.html",
        _library_ctx(request, q, current_slug=slug),
    )


@router.get("/api/library/tree", response_class=HTMLResponse)
async def api_library_tree(
    request: Request,
    q: str = "",
    slug: str | None = Depends(film_slug_query),
) -> HTMLResponse:
    """Return the Mojica LeftPane body for HTMX filter swaps.

    Targeted by ``.ch-lp .filter input`` via ``hx-target=".ch-lp .scroll"
    hx-swap="innerHTML"``. The response is the inner fragment of the
    scrolling region — films + collections + shared — wrapped by the
    enclosing ``_left_pane.html`` on the initial render. The new
    endpoint avoids breaking the legacy ``/api/library/filter`` contract
    (still in use by the v0.3 sidebar inside ``.ch-main``).
    """
    return templates.TemplateResponse(
        request,
        "partials/_left_pane_body.html",
        _chrome_filter_ctx(request, q, current_slug=slug),
    )
