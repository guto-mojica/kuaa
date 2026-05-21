"""Library sidebar routes — registry filter (legacy + Mojica chrome) + management.

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

Routes
------
GET  /api/library/filter              — filtered library tree (HTMX fragment)
GET  /api/library/tree                — Mojica LeftPane body (HTMX fragment)
GET  /api/library/select/{slug}       — navigate to a film (HX-Redirect)
GET  /api/library/add-form            — inline add-film form
POST /api/library/add                 — register a new film
GET  /api/library/remove-confirm/{slug} — inline remove confirmation
POST /api/library/remove/{slug}       — deregister (+ optional data wipe)
"""
from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, Response

from api.deps import film_slug_query, get_config, make_ctx
from api.services.chrome_service import build_chrome_context
from api.templates import templates

router = APIRouter()


def _library_ctx(request: Request, q: str = "", current_slug: str | None = None) -> dict:
    """Build the legacy sidebar context: global state + filtered registry film list."""
    from cinemateca.library import library_state, scan_library

    cfg = get_config()
    library_dir = Path(cfg.paths.library_dir)

    films = scan_library(library_dir)
    if q.strip():
        needle = q.strip().lower()
        films = [f for f in films if needle in f.title.lower() or needle in f.slug.lower()]

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


def _tree_response(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "partials/library_tree.html",
        _library_ctx(request),
    )


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


@router.get("/api/library/select/{slug}")
async def api_library_select(slug: str) -> Response:
    """Navigate the browser to the film's scenes tab via HX-Redirect."""
    return Response(status_code=200, headers={"HX-Redirect": f"/scenes?film={slug}"})


@router.get("/api/library/add-form", response_class=HTMLResponse)
async def api_library_add_form(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request, "partials/add_film_form.html", {"request": request}
    )


@router.post("/api/library/add", response_class=HTMLResponse)
async def api_library_add(
    request: Request,
    video_path: str = Form(...),
    title: str = Form(default=""),
) -> HTMLResponse:
    from cinemateca.library import register_film
    from cinemateca.pipeline import slugify

    cfg = get_config()
    library_dir = Path(cfg.paths.library_dir)
    video = Path(video_path.strip()).expanduser()

    # Accept a bare filename — resolve against the raw dir.
    if not video.is_absolute():
        candidate = Path(cfg.paths.raw_dir) / video
        if candidate.exists():
            video = candidate

    if not video.exists():
        ctx = {"request": request, "error": f"File not found: {video_path}"}
        return templates.TemplateResponse(
            request, "partials/add_film_form.html", ctx
        )

    slug = slugify(video.stem)
    film_title = title.strip() or video.stem.replace("_", " ").title()

    try:
        register_film(
            library_dir,
            slug=slug,
            title=film_title,
            year=None,
            raw_filename=video.name,
        )
    except ValueError as exc:
        ctx = {"request": request, "error": str(exc)}
        return templates.TemplateResponse(
            request, "partials/add_film_form.html", ctx
        )

    # Symlink the source file into the per-film raw/ dir so the pipeline
    # can find it regardless of where the original lives on disk.
    raw_dir = library_dir / slug / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    link = raw_dir / video.name
    if not link.exists() and not link.is_symlink():
        link.symlink_to(video.resolve())

    return _tree_response(request)


@router.get("/api/library/remove-confirm/{slug}", response_class=HTMLResponse)
async def api_library_remove_confirm(request: Request, slug: str) -> HTMLResponse:
    from cinemateca.library import load_registry

    cfg = get_config()
    registry = load_registry(cfg.paths.library_dir)
    film_title = registry.get(slug, {}).get("title", slug)
    return templates.TemplateResponse(
        request,
        "partials/remove_film_confirm.html",
        {"request": request, "slug": slug, "film_title": film_title},
    )


@router.post("/api/library/remove/{slug}", response_class=HTMLResponse)
async def api_library_remove(
    request: Request,
    slug: str,
    wipe: str = Form(default=""),
) -> HTMLResponse:
    from cinemateca.library import delete_film, load_registry, save_registry  # noqa: F401

    cfg = get_config()
    library_dir = Path(cfg.paths.library_dir)

    registry = load_registry(library_dir)
    if slug in registry:
        delete_film(library_dir, slug)

    if wipe:
        film_dir = library_dir / slug
        if film_dir.exists():
            shutil.rmtree(film_dir)

    return _tree_response(request)
