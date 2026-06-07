"""Library routes: filter / tree / select / add-form / add / remove-confirm / remove."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, Response

from api.deps import film_slug_query, get_config, make_ctx, request_gettext, toast_trigger
from api.services.library_admin import register_and_symlink, resolve_video_path
from api.services.processing_render import processing_tab_response
from api.services.library_render import chrome_filter_ctx, library_ctx
from api.templates import templates
from cinemateca.errors import IndexMissing

router = APIRouter()


@router.get("/api/library/filter", response_class=HTMLResponse)
async def api_library_filter(
    request: Request, q: str = "", slug: str | None = Depends(film_slug_query)
) -> HTMLResponse:
    return templates.TemplateResponse(
        request, "partials/library_tree.html", library_ctx(request, q, current_slug=slug)
    )


@router.get(
    "/api/library/tree", response_class=HTMLResponse
)  # Mojica LeftPane body for HTMX filter swaps
async def api_library_tree(
    request: Request, q: str = "", slug: str | None = Depends(film_slug_query)
) -> HTMLResponse:
    return templates.TemplateResponse(
        request, "partials/_left_pane_body.html", chrome_filter_ctx(request, q, current_slug=slug)
    )


@router.get("/api/library/select/{slug}")
async def api_library_select(
    slug: str,
) -> Response:  # HX-Redirect → /scenes?film=; IndexMissing on unknown slug
    from cinemateca.library import load_registry

    cfg = get_config()
    registry = load_registry(cfg.paths.library_dir)
    if slug not in registry:
        raise IndexMissing(f"unknown film: {slug!r}")
    return Response(status_code=200, headers={"HX-Redirect": f"/scenes?film={slug}"})


@router.get("/api/library/add-form", response_class=HTMLResponse)
async def api_library_add_form(request: Request, source: str = "left-pane") -> HTMLResponse:
    return templates.TemplateResponse(
        request, "partials/add_film_form.html", make_ctx(request, source=source)
    )


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
    _ = request_gettext(request)

    _ERROR_MESSAGES = {
        "video_not_found": _("File not found. Check the path or filename."),
        "slug_duplicate": _("A film with this name is already in the library."),
        "already_in_library": _("This film is already in the library."),
    }

    def _error(error_key: str, sub: str = "") -> HTMLResponse | Response:
        if source == "processing":
            return processing_tab_response(request, error_message=_ERROR_MESSAGES.get(error_key, error_key), sub=sub)
        ctx = make_ctx(request, error_key=error_key)
        return templates.TemplateResponse(request, "partials/add_film_form.html", ctx)

    if not video.exists():
        return _error("video_not_found", str(video))

    slug = slugify(video.stem)
    film_title = title.strip() or video.stem.replace("_", " ").title()

    try:
        register_and_symlink(library_dir, video, slug, film_title)
    except (
        ValueError
    ):  # slug duplicate — block only if raw symlink is healthy; repair orphans silently
        from cinemateca.library import scan_library

        existing = next((f for f in scan_library(library_dir) if f.slug == slug), None)
        if existing and existing.raw_path.exists():
            return _error("already_in_library", slug)
        per_film_raw = library_dir / slug / "raw"
        per_film_raw.mkdir(parents=True, exist_ok=True)
        link = per_film_raw / video.name
        if not link.is_symlink():
            link.symlink_to(video.resolve())

    if source == "processing":
        resp: HTMLResponse | Response = processing_tab_response(request, new_slug=slug)
        toast_trigger(resp, title=_("Film added"), sub=film_title, kind="success")
        return resp

    from api.jobs import STEP_DEFS, queue_job  # enqueue and redirect to Processing tab

    symlink_path = library_dir / slug / "raw" / video.name
    queue_job(str(symlink_path), {name for name, _ in STEP_DEFS}, cfg)
    resp = Response(status_code=200, headers={"HX-Redirect": f"/processing?film={slug}"})
    return resp


@router.get("/api/library/remove-confirm/{slug}", response_class=HTMLResponse)
async def api_library_remove_confirm(request: Request, slug: str) -> HTMLResponse:
    from cinemateca.library import load_registry

    cfg = get_config()
    registry = load_registry(cfg.paths.library_dir)
    film_title = registry.get(slug, {}).get("title", slug)
    return templates.TemplateResponse(
        request,
        "partials/remove_film_confirm.html",
        {**make_ctx(request), "slug": slug, "film_title": film_title},
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
    return templates.TemplateResponse(
        request, "partials/_left_pane_body.html", chrome_filter_ctx(request, "")
    )
