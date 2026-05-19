"""Library sidebar routes — per-film inventory, selection, and registration."""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse

from api.deps import get_config, make_ctx
from api.templates import templates

logger = logging.getLogger(__name__)
router = APIRouter()

_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".m4v", ".webm"}


def _library_ctx(request: Request, q: str = "") -> dict:
    """Build the sidebar context: per-film state + filtered inventory."""
    from cinemateca.library import library_state, scan_library

    cfg = get_config()
    raw_dir = Path(cfg.paths.raw_dir)
    metadata_dir = Path(cfg.paths.metadata_dir)
    films_dir = Path(cfg.paths.data_dir) / "films"

    films = scan_library(raw_dir=raw_dir, metadata_dir=metadata_dir, films_dir=films_dir)
    if q.strip():
        needle = q.strip().lower()
        films = [f for f in films if needle in f.title.lower() or needle in f.slug.lower()]

    state = library_state(
        raw_dir=raw_dir,
        metadata_dir=metadata_dir,
        embeddings_index_path=Path(cfg.paths.embeddings_dir) / cfg.embeddings.filename,
    )
    return make_ctx(request, films=films, library_state=state)


def _register_film(cfg, vp: Path, title: str = "") -> None:
    """Create per-film dirs and film.json. Idempotent (exist_ok)."""
    slug = vp.stem.lower().replace(" ", "_")
    film_title = title.strip() or vp.stem.replace("_", " ").title()
    film_dir = Path(cfg.paths.data_dir) / "films" / slug
    (film_dir / "metadata").mkdir(parents=True, exist_ok=True)
    (film_dir / "frames").mkdir(parents=True, exist_ok=True)
    (film_dir / "embeddings").mkdir(parents=True, exist_ok=True)
    film_dir.joinpath("film.json").write_text(
        json.dumps({"slug": slug, "title": film_title, "raw_path": str(vp)},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Registered film %r at %s", slug, vp)


def _form_error(request: Request, msg: str) -> HTMLResponse:
    """Return the add-film form retargeted to #add-film-zone with an error."""
    resp = templates.TemplateResponse(
        request,
        "partials/add_film_form.html",
        make_ctx(request, error=msg),
    )
    resp.headers["HX-Retarget"] = "#add-film-zone"
    resp.headers["HX-Reswap"] = "innerHTML"
    return resp


@router.get("/api/library/filter", response_class=HTMLResponse)
async def api_library_filter(request: Request, q: str = "") -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "partials/library_tree.html",
        _library_ctx(request, q),
    )


@router.get("/api/library/select/{slug}", response_class=HTMLResponse)
async def api_library_select(request: Request, slug: str) -> HTMLResponse:
    """Set the active-film cookie and reload the page."""
    response = HTMLResponse("")
    response.set_cookie(
        "active_film", slug, max_age=86400 * 365, httponly=True, samesite="lax"
    )
    referer = request.headers.get("referer", "/")
    response.headers["HX-Redirect"] = referer
    return response


@router.get("/api/library/add-form", response_class=HTMLResponse)
async def api_library_add_form(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "partials/add_film_form.html",
        make_ctx(request),
    )


@router.get("/api/library/cancel-add", response_class=HTMLResponse)
async def api_library_cancel_add(request: Request) -> HTMLResponse:
    return HTMLResponse("")


@router.get("/api/library/pick-file", response_class=HTMLResponse)
async def api_library_pick_file(request: Request) -> HTMLResponse:
    """Open a macOS native file-picker dialog and register the chosen film.

    Runs ``osascript`` asynchronously (the browser tab shows a loading
    spinner while the dialog is open).  On cancel or failure the form is
    restored; on success the library tree refreshes with the new entry.
    """
    script = (
        "POSIX path of (choose file "
        'with prompt "Select video file" '
        'of type {"public.movie"})'
    )
    try:
        proc = await asyncio.create_subprocess_exec(
            "osascript", "-e", script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
    except FileNotFoundError:
        return _form_error(request, "File picker unavailable (osascript not found).")

    if proc.returncode != 0:
        # User pressed Cancel — silently restore the tree.
        return templates.TemplateResponse(
            request, "partials/library_tree.html", _library_ctx(request)
        )

    vp = Path(stdout.decode().strip())
    if not vp.is_file() or vp.suffix.lower() not in _VIDEO_EXTENSIONS:
        return _form_error(
            request,
            f"Unsupported format: {vp.suffix or '(no extension)'}",
        )

    cfg = get_config()
    _register_film(cfg, vp)
    return templates.TemplateResponse(
        request, "partials/library_tree.html", _library_ctx(request)
    )


@router.post("/api/library/add", response_class=HTMLResponse)
async def api_library_add(
    request: Request,
    video_path: str = Form(...),
    title: str = Form(default=""),
) -> HTMLResponse:
    """Register a film from a typed/pasted path."""
    cfg = get_config()
    vp = Path(video_path.strip()).expanduser()

    # Accept bare filenames relative to raw_dir.
    if not vp.is_file():
        candidate = Path(cfg.paths.raw_dir) / vp.name
        if candidate.is_file():
            vp = candidate

    if not vp.is_file() or vp.suffix.lower() not in _VIDEO_EXTENSIONS:
        return _form_error(
            request,
            f"File not found or unsupported format: {vp.name or video_path}",
        )

    _register_film(cfg, vp, title)
    return templates.TemplateResponse(
        request, "partials/library_tree.html", _library_ctx(request)
    )
