"""Library sidebar routes — registry management.

Shows the registry-backed film list filtered by name.
Per-film scene counts and processed state are REAL (read from
``<library_dir>/<slug>/metadata/keyframes_metadata.json``).

Routes
------
GET  /api/library/filter              — filtered library tree (HTMX fragment)
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
from api.templates import templates

router = APIRouter()


# ── helpers ───────────────────────────────────────────────────────────────────

def _library_ctx(request: Request, q: str = "", current_slug: str | None = None) -> dict:
    from cinemateca.library import library_state, scan_library

    cfg = get_config()
    library_dir = Path(cfg.paths.library_dir)
    films = scan_library(library_dir)
    if q.strip():
        needle = q.strip().lower()
        films = [f for f in films if needle in f.title.lower() or needle in f.slug.lower()]
    state = library_state(library_dir)
    return make_ctx(request, films=films, library_state=state, current_slug=current_slug)


def _tree_response(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "partials/library_tree.html",
        _library_ctx(request),
    )


# ── routes ────────────────────────────────────────────────────────────────────

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
    video = Path(video_path.strip())

    # Accept a bare filename — resolve against the raw dir.
    if not video.is_absolute():
        candidate = Path(cfg.paths.data_dir) / "raw" / video
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
    from cinemateca.library import delete_film, load_registry, save_registry

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
