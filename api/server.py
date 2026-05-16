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
from api.routes import about, annotate, library, processing, scenes, search, tabs
from api.services.catalog import build_scenes_context
from api.services.film_context import FilmContext
from api.templates import templates

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_config()
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


# Each tab's full context is built by the SAME code path the matching
# `/tab/<x>` route uses, so a direct full-page GET renders identical tab
# markup (modulo the surrounding base chrome).
#
# `scenes` is the Phase-3a extracted case: its builder now lives in the
# catalog SERVICE and takes a `FilmContext`, so the `/tab/scenes` route
# and this full-page path both call `build_scenes_context(FilmContext
# .from_config(...))` — same service function, same context keys. The
# other three builders still live in their route modules (Phase 3b/3c/4
# extract them) and remain zero-arg.
_TAB_CONTEXT_BUILDERS = {
    "search": search.build_search_context,
    "scenes": lambda: build_scenes_context(FilmContext.from_config(get_config())),
    "annotate": annotate.build_annotate_context,
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
    from cinemateca.library import scan_library

    # Base scan feeds the sidebar library tree. The search/scenes/annotate tab
    # builders do not re-supply `films`, so this scan is the only source for
    # their sidebar. Do NOT remove it: the `processing` builder intentionally
    # overrides `films` (see the merge below), but the other three depend on
    # this scan. On /processing both scans run -- a known, harmless redundancy.
    #
    # Phase-3a decision: this double scan is NOT de-duplicated here. The
    # catalog service / FilmContext extracted in Phase 3a is single-film/
    # global and deliberately does NOT model the library FILM LIST (that is
    # `scan_library`, owned by the Phase-5 library / per-film data model).
    # Routing the sidebar scan through a shared object would require that
    # Phase-5 abstraction; folding it in now would be scope bleed and would
    # entangle this refactor with an unresolved product decision. Left as-is
    # intentionally; Phase 5 owns collapsing it.
    films = scan_library(
        raw_dir=Path(cfg.paths.raw_dir),
        metadata_dir=Path(cfg.paths.metadata_dir),
    )
    base_ctx = {
        "active_tab": active_tab,
        "processing_jobs": 0,
        "films": films,
        "selected_slug": films[0].slug if films else None,
    }
    # `{**base_ctx, **tab_ctx}`: tab_ctx wins on key collisions. The
    # `processing` builder deliberately re-supplies `films`, overriding the
    # base value here; that override is intended, not a bug.
    tab_ctx = _TAB_CONTEXT_BUILDERS[active_tab]()
    return templates.TemplateResponse(
        request,
        "base.html",
        make_ctx(request, **{**base_ctx, **tab_ctx}),
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
