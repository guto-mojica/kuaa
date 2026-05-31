"""Library sidebar routes — registry filter (legacy + Mojica chrome) + management.

Render/context helpers live in :mod:`api.services.library_render` (A2 Task 5).
Admin orchestration (register/symlink/remove) lives in
:mod:`api.services.library_admin` (A2 Task 5). The route decorators below are
the authoritative path list (filter / tree / select / add-form / add /
remove-confirm / remove).
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, Response

from api.deps import film_slug_query, get_config, make_ctx
from api.services.library_admin import register_and_symlink, resolve_video_path
from api.services.library_render import chrome_filter_ctx, library_ctx, tree_response
from api.templates import templates
from cinemateca.errors import IndexMissing

router = APIRouter()


@router.get("/api/library/filter", response_class=HTMLResponse)
async def api_library_filter(
    request: Request,
    q: str = "",
    slug: str | None = Depends(film_slug_query),
) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "partials/library_tree.html",
        library_ctx(request, q, current_slug=slug),
    )


@router.get("/api/library/tree", response_class=HTMLResponse)
async def api_library_tree(
    request: Request,
    q: str = "",
    slug: str | None = Depends(film_slug_query),
) -> HTMLResponse:
    """Return the Mojica LeftPane body for HTMX filter swaps."""
    return templates.TemplateResponse(
        request,
        "partials/_left_pane_body.html",
        chrome_filter_ctx(request, q, current_slug=slug),
    )


@router.get("/api/library/select/{slug}")
async def api_library_select(slug: str) -> Response:
    """Navigate the browser to the film's scenes tab via HX-Redirect.

    Validates the slug against the films.json registry before redirecting.
    An unknown slug raises :class:`IndexMissing` (→ 404) rather than
    silently redirecting to a non-existent film page.
    """
    from cinemateca.library import load_registry

    cfg = get_config()
    registry = load_registry(cfg.paths.library_dir)
    if slug not in registry:
        raise IndexMissing(f"unknown film: {slug!r}")
    return Response(status_code=200, headers={"HX-Redirect": f"/scenes?film={slug}"})


@router.get("/api/library/add-form", response_class=HTMLResponse)
async def api_library_add_form(request: Request) -> HTMLResponse:
    # make_ctx (not a bare {"request": …}) so the form's {{ _(...) }} strings
    # honour the locale cookie instead of the global default.
    return templates.TemplateResponse(request, "partials/add_film_form.html", make_ctx(request))


@router.post("/api/library/add", response_class=HTMLResponse, response_model=None)
async def api_library_add(
    request: Request,
    video_path: str = Form(...),
    title: str = Form(default=""),
    source: str = Form(default=""),
) -> HTMLResponse | Response:
    from cinemateca.pipeline import slugify

    cfg = get_config()
    library_dir = Path(cfg.paths.library_dir)
    raw_dir = Path(cfg.paths.raw_dir)

    video = resolve_video_path(video_path, str(raw_dir))
    # On failure, re-render the form with a U1 accessible inline error keyed by
    # ``error_key`` (the partial translates it + sets aria-invalid). make_ctx
    # carries the locale so the message + labels render in the cookie locale.
    if not video.exists():
        ctx = make_ctx(request, error_key="video_not_found")
        return templates.TemplateResponse(request, "partials/add_film_form.html", ctx)

    slug = slugify(video.stem)
    film_title = title.strip() or video.stem.replace("_", " ").title()

    try:
        register_and_symlink(library_dir, video, slug, film_title, raw_dir)
    except ValueError:  # register_film raises only for an already-registered slug
        ctx = make_ctx(request, error_key="slug_duplicate")
        return templates.TemplateResponse(request, "partials/add_film_form.html", ctx)

    if source == "processing":
        return Response(
            status_code=200,
            headers={"HX-Redirect": f"/processing?film={slug}"},
        )

    return templates.TemplateResponse(
        request,
        "partials/_left_pane_body.html",
        chrome_filter_ctx(request, ""),
    )


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
    from api.services.library_admin import remove_film_and_wipe

    cfg = get_config()
    library_dir = Path(cfg.paths.library_dir)
    remove_film_and_wipe(library_dir, slug, wipe=bool(wipe))
    return tree_response(request)
