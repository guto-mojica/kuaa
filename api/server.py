"""FastAPI application — mounted by uvicorn via app.py."""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Safety net for non-installed dev environments
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from api.deps import get_config, make_ctx
from api.routes import about, annotate, export, library, processing, scenes, search, tabs
from api.services.annotations import build_annotate_context
from api.services.catalog import build_scenes_context_aggregate
from api.services.film_context import FilmContext
from api.templates import templates

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_config()
    # CLI paths call setup_logging in their entrypoints; the FastAPI app
    # never did, so config.logging.* (level + to_file + filename) was
    # silently ignored under uvicorn and only the default stream handler
    # ran. Wire it here so /logs/cinemateca.log gets web-app events too.
    from cinemateca.config import setup_logging

    setup_logging(cfg)
    data_dir = Path(cfg.paths.data_dir).resolve()
    if data_dir.exists():
        app.mount("/media", StaticFiles(directory=str(data_dir)), name="media")
        logger.info("Serving media from %s", data_dir)
    else:
        logger.warning("data_dir not found — keyframe images will not be served: %s", data_dir)
    yield


_BASE = Path(__file__).parent.parent

app = FastAPI(title="Cinemateca AI", version="0.3.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(_BASE / "web" / "static")), name="static")

app.include_router(tabs.router)
app.include_router(search.router)
app.include_router(scenes.router)
app.include_router(annotate.router)
app.include_router(processing.router)
app.include_router(about.router)
app.include_router(library.router)
app.include_router(export.router)


# Each tab's full context is built by the SAME code path the matching
# `/tab/<x>` route uses, so a direct full-page GET renders identical tab
# markup (modulo the surrounding base chrome).
#
# `scenes` and `annotate` are the Phase-3a/3b extracted cases: their
# builders now live in SERVICES and take a `FilmContext`, so the
# `/tab/<x>` route and this full-page path both call the same service
# function with `FilmContext.from_config(...)` — same context keys. The
# other two builders still live in their route modules (Phase 3c/4
# extract them) and remain zero-arg.
_TAB_CONTEXT_BUILDERS = {
    # ``"search"`` is intentionally not in this map — its builder is
    # slug-aware (per-film vs aggregate tag vocabulary), so render_page
    # calls ``search.build_search_context(current_slug)`` directly after
    # parsing ``?film=<slug>``.
    # Full-page /scenes uses the same aggregate builder as /tab/scenes (no slug
    # → aggregate across all films). Previously used FilmContext.from_config
    # which reads the FLAT metadata_dir, breaking Phase-1a parity with the
    # HTMX tab path after T9 introduced library-tree routing.
    "scenes": lambda: build_scenes_context_aggregate(get_config()),
    # Annotate stays single-film (from_config) intentionally: an aggregate
    # annotate view (write-path, scene-by-scene editing across all films) is
    # deferred to a later plan (T9 docstring). /tab/annotate with slug=None also
    # uses from_config, so /annotate full-page and /tab/annotate are consistent.
    "annotate": lambda: build_annotate_context(FilmContext.from_config(get_config())),
    "processing": processing.build_processing_context,
}


def render_page(request: Request, active_tab: str) -> HTMLResponse:
    """Render a full page with base chrome + the active tab's full context.

    Builds the base context (library tree, processing badge) and merges
    in the active tab's context via that tab's shared builder, so the
    included partial in ``base.html`` receives exactly the same variables
    it would as a standalone ``/tab/<x>`` fragment.
    """
    cfg = get_config()
    from cinemateca.library import library_state, scan_library

    # Base scan feeds the sidebar registry + global state. The
    # search/scenes/annotate tab builders do not re-supply `films`, so this
    # is the only source for their sidebar. Do NOT remove it: the
    # `processing` builder intentionally overrides `films` (see the merge
    # below), but the other three depend on this scan.
    #
    # scan_library now reads films.json (registry) and returns real per-film
    # scene_count/is_processed. The processing builder re-supplies `films`
    # from the same source; collapsing the double-scan into one request-
    # scoped library object belongs to T9/T10, not here.
    library_dir = Path(cfg.paths.library_dir)
    films = scan_library(library_dir)
    state = library_state(library_dir)
    base_ctx = {
        "active_tab": active_tab,
        "processing_jobs": 0,
        "films": films,
        "library_state": state,
    }
    # HTMX-driven film switches issue full-page GETs with ?film=<slug>
    # (the selector's hx-push-url propagates the slug into the URL bar),
    # so render_page must read it back so the sidebar selector keeps
    # the right option marked selected on the response. Read it BEFORE
    # building tab_ctx so slug-aware builders (search) can scope their
    # tag vocabulary to the active film.
    current_slug = request.query_params.get("film") or None
    # `{**base_ctx, **tab_ctx}`: tab_ctx wins on key collisions. The
    # `processing` builder deliberately re-supplies `films`, overriding the
    # base value here; that override is intended, not a bug.
    if active_tab == "search":
        tab_ctx = search.build_search_context(current_slug)
    elif active_tab == "scenes" and current_slug:
        from api.services.catalog import build_scenes_context
        tab_ctx = build_scenes_context(FilmContext.for_film(cfg, current_slug))
    else:
        tab_ctx = _TAB_CONTEXT_BUILDERS[active_tab]()
    return templates.TemplateResponse(
        request,
        "base.html",
        make_ctx(request, current_slug=current_slug, **{**base_ctx, **tab_ctx}),
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return render_page(request, "search")


@app.get("/search", response_class=HTMLResponse)
async def page_search(request: Request) -> HTMLResponse:
    return render_page(request, "search")


@app.get("/scenes", response_class=HTMLResponse)
async def page_scenes(request: Request) -> HTMLResponse:
    return render_page(request, "scenes")


@app.get("/annotate", response_class=HTMLResponse)
async def page_annotate(request: Request) -> HTMLResponse:
    return render_page(request, "annotate")


@app.get("/processing", response_class=HTMLResponse)
async def page_processing(request: Request) -> HTMLResponse:
    return render_page(request, "processing")
