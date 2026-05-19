"""Library sidebar routes — per-film inventory, selection, and registration."""
from __future__ import annotations

import json
import logging
import shutil
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


@router.post("/api/library/add", response_class=HTMLResponse)
async def api_library_add(
    request: Request,
    video_path: str = Form(...),
    title: str = Form(default=""),
) -> HTMLResponse:
    """Register a film from a typed or pasted path.

    Only the file extension is validated at registration time — the file
    does not need to be currently accessible (e.g. path to an external
    drive that may be unmounted).  A bare filename (no directory
    component) is resolved against ``raw_dir`` as a convenience.
    """
    cfg = get_config()
    vp = Path(video_path.strip()).expanduser()

    # Bare filename → try raw_dir
    if len(vp.parts) == 1:
        candidate = Path(cfg.paths.raw_dir) / vp
        if candidate.exists() or candidate.is_symlink():
            vp = candidate

    if not vp.suffix or vp.suffix.lower() not in _VIDEO_EXTENSIONS:
        return _form_error(
            request,
            f"Unsupported format: {vp.suffix or '(no extension)'} — "
            f"accepted: {', '.join(sorted(_VIDEO_EXTENSIONS))}",
        )

    _register_film(cfg, vp, title)
    return templates.TemplateResponse(
        request, "partials/library_tree.html", _library_ctx(request)
    )


@router.get("/api/library/remove-confirm/{slug}", response_class=HTMLResponse)
async def api_library_remove_confirm(request: Request, slug: str) -> HTMLResponse:
    """Return an inline confirmation strip that replaces the film row."""
    cfg = get_config()
    film_json = Path(cfg.paths.data_dir) / "films" / slug / "film.json"
    title = slug.replace("_", " ").title()
    if film_json.exists():
        try:
            meta = json.loads(film_json.read_text(encoding="utf-8"))
            title = meta.get("title", title)
        except (json.JSONDecodeError, OSError):
            pass
    return templates.TemplateResponse(
        request,
        "partials/remove_film_confirm.html",
        make_ctx(request, slug=slug, film_title=title),
    )


@router.post("/api/library/remove/{slug}", response_class=HTMLResponse)
async def api_library_remove(
    request: Request,
    slug: str,
    wipe: str = Form(default=""),
) -> HTMLResponse:
    """Remove a film entry.  If ``wipe`` is non-empty, delete the full
    per-film data directory; otherwise just remove ``film.json`` so the
    processed artefacts are kept but the film no longer appears in the
    library list."""
    cfg = get_config()
    film_dir = Path(cfg.paths.data_dir) / "films" / slug

    if wipe:
        if film_dir.exists():
            shutil.rmtree(film_dir)
            logger.info("Wiped film dir for %r", slug)
    else:
        film_json = film_dir / "film.json"
        if film_json.exists():
            film_json.unlink()
            logger.info("Removed film.json for %r (data kept)", slug)

    return templates.TemplateResponse(
        request, "partials/library_tree.html", _library_ctx(request)
    )
