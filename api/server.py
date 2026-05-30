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
from api.error_handlers import install_error_handlers
from api.middleware import RequestContextMiddleware
from api.routes import (
    about,
    annotate,
    export,
    library,
    palette,
    processing,
    rimas,
    scenes,
    search,
    system,
)
from api.routes import (
    eval as eval_routes,
)
from api.services.annotations import build_annotate_context, normalize_annotate_tab
from api.services.chrome_service import build_chrome_context
from api.services.processing_render import build_processing_context
from api.services.rhymes_service import build_rimas_context
from api.services.scenes import build_cenas_context, build_timeline_context
from api.templates import templates
from cinemateca.library import FilmContext, keyframe_url, load_json, scan_library

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

_OPENAPI_TAGS = [
    {
        "name": "search",
        "description": (
            "Semantic search over the film archive. Supports text (CLIP), "
            "image upload, audio (CLAP), and cross-modal fusion queries."
        ),
    },
    {
        "name": "library",
        "description": "Film registration, selection, and library-tree management.",
    },
    {
        "name": "scenes",
        "description": "Scene browsing, filtering, and per-scene inspector fragments.",
    },
    {
        "name": "annotate",
        "description": "Manual tag curation and scene description editing.",
    },
    {
        "name": "rimas",
        "description": "Cross-film visual-rhyme discovery (Rimas Visuais).",
    },
    {
        "name": "processing",
        "description": "Pipeline job control — ingest and process video files.",
    },
    {
        "name": "export",
        "description": "Structured catalog exports (JSON / CSV).",
    },
    {
        "name": "eval",
        "description": (
            "Eval-set grading and retrieval-quality metrics (admin-gated by "
            "``EVAL_ADMIN_TOKEN``)."
        ),
    },
    {
        "name": "system",
        "description": "Health, readiness, and operational endpoints.",
    },
]

app = FastAPI(
    title="Cinemateca AI",
    version="0.3.0",
    description=(
        "Offline audiovisual cataloguing API for film archives. "
        "Indexes scenes from video files and exposes semantic search "
        "(text, image, audio, fusion) over CLIP and CLAP embeddings, "
        "with hybrid BM25+vector retrieval and cross-encoder reranking. "
        "All inference runs locally — no cloud APIs are used."
    ),
    openapi_tags=_OPENAPI_TAGS,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)
app.add_middleware(RequestContextMiddleware)
app.mount("/static", StaticFiles(directory=str(_BASE / "web" / "static")), name="static")

app.include_router(search.router, tags=["search"])
app.include_router(scenes.router, tags=["scenes"])
app.include_router(annotate.router, tags=["annotate"])
app.include_router(processing.router, tags=["processing"])
app.include_router(about.router, tags=["system"])
app.include_router(library.router, tags=["library"])
app.include_router(export.router, tags=["export"])
app.include_router(rimas.router, tags=["rimas"])
app.include_router(palette.router, tags=["system"])
app.include_router(eval_routes.router, tags=["eval"])
app.include_router(system.router)

install_error_handlers(app)

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
    # ``"search"``, ``"scenes"`` and ``"rimas"`` are intentionally NOT in
    # this map — their builders are slug-aware (per-film vs aggregate),
    # so render_page calls them directly with ``current_slug`` after
    # parsing ``?film=<slug>``. See the matching ``elif`` branches below.
    # annotate is handled directly in render_page's if/elif chain so it can
    # receive current_slug (from ?film= query param or active_film cookie).
    "processing": build_processing_context,
    # Rimas Visuais (cross-film visual rhymes) — Task 21 wires the real
    # service builder. The full-page render reads ``?anchor=`` like the
    # tab-fragment endpoint so a deep-share URL (``/rimas?anchor=jeca/1``)
    # lands directly on the requested scene. ``anchor=None`` falls back
    # to the default-anchor branch inside the service.
    "rimas": lambda: build_rimas_context(get_config(), anchor=None),
}


# Mojica chrome metadata per tab. Maps the internal EN tab key used in URLs +
# Python (search/scenes/annotate/processing/rimas) to:
#   - active_tab (PT slug used by ``[data-active-tab]`` on <body>),
#   - compact_lp (Anotar collapses the LeftPane to maximise the work surface),
#   - has_right_pane (every Phase-1 tab keeps a right pane; templates that don't
#     populate it render an empty <aside> — semantically harmless).
# Kept here (not in deps.py) because it concerns route-level page composition,
# not per-request locale or film context. Phase 2+ tasks may move some of this
# into the page templates themselves once they extend base.html directly.
def build_home_context(cfg: object) -> dict:
    """Return film-card list for the home library-overview page.

    Each card carries the Film object plus the URL of its first keyframe
    (or None when the film has not been processed yet).
    """
    library_dir = Path(cfg.paths.library_dir)
    films = list(scan_library(library_dir))
    film_cards = []
    for film in films:
        thumbnail_url = None
        try:
            from cinemateca.library import FilmContext as _FC

            ctx = _FC.for_film(cfg, film.slug)
            kf_meta = load_json(ctx.metadata_dir / "keyframes_metadata.json") or []
            if isinstance(kf_meta, list) and kf_meta:
                thumbnail_url = keyframe_url(kf_meta[0].get("filepath", ""), ctx.data_dir)
        except Exception:
            pass
        film_cards.append({"film": film, "thumbnail_url": thumbnail_url})
    return {"film_cards": film_cards}


_TAB_CHROME = {
    # has_right_pane=False for every tab: each tab manages its own right pane
    # inside .tab-panel (via .b-rp, .c-rp, .r-rp, etc.). Setting True would add
    # an empty ch-right 380px grid column AND a duplicate id="right-pane" element,
    # breaking HTMX targeting and stealing layout space from ch-main.
    "home": {"active_tab": "home", "compact_lp": False, "has_right_pane": False},
    "search": {"active_tab": "buscar", "compact_lp": False, "has_right_pane": False},
    "scenes": {"active_tab": "cenas", "compact_lp": False, "has_right_pane": False},
    "annotate": {"active_tab": "anotar", "compact_lp": True, "has_right_pane": False},
    # NOTE: the body's data-active-tab uses the short slug "proc" (not the
    # full PT "processamento") so the topbar tab chip's `data-tab="proc"`
    # selector matches in CSS / JS. Task 7 wired this contract.
    "processing": {"active_tab": "proc", "compact_lp": False, "has_right_pane": False},
    "rimas": {"active_tab": "rimas", "compact_lp": False, "has_right_pane": False},
}


def render_page(request: Request, active_tab: str) -> HTMLResponse:
    """Render a full page with base chrome + the active tab's full context.

    Builds the base context (library tree, processing badge) and merges
    in the active tab's context via that tab's shared builder, so the
    included partial in ``base.html`` receives exactly the same variables
    it would as a standalone ``/tab/<x>`` fragment.
    """
    cfg = get_config()

    # HTMX-driven film switches issue full-page GETs with ?film=<slug>
    # (the selector's hx-push-url propagates the slug into the URL bar),
    # so render_page must read it back so the sidebar selector keeps
    # the right option marked selected on the response. Read it BEFORE
    # building tab_ctx so slug-aware builders (search) can scope their
    # tag vocabulary to the active film, AND before building the chrome
    # context so the LeftPane marks the .ch-film.active row correctly.
    _raw_slug = request.query_params.get("film") or request.cookies.get("active_film") or None
    # Normalise to lowercase (all registered slugs are lowercase via slugify)
    # and validate against the library directory so a stale cookie or wrong-
    # cased slug doesn't propagate a ValueError into every service call.
    if _raw_slug:
        _raw_slug = _raw_slug.lower()
        _film_dir = Path(cfg.paths.library_dir) / _raw_slug
        current_slug: str | None = _raw_slug if _film_dir.is_dir() else None
    else:
        current_slug = None

    # Task-8: lift the per-request chrome bag (films, library_state,
    # active_job_slugs/count, total_runtime, collections, viewers, …)
    # into a single builder. This replaces the inline scan_library /
    # library_state pair and adds the Mojica-chrome keys the new
    # LeftPane + IconRail + TopBar need. The keys ``films`` and
    # ``library_state`` are still carried by ``chrome_ctx`` so legacy
    # tab partials (and the still-wrapped legacy sidebar inside
    # .ch-main) see the same values they did before. The ``processing``
    # builder still re-supplies ``films`` downstream — that override is
    # intentional (see merge below), not a bug.
    bucket_param = request.query_params.get("bucket") or None
    chrome_ctx = build_chrome_context(cfg, current_slug=current_slug, current_bucket=bucket_param)

    base_ctx = {
        "active_tab": active_tab,
        "processing_jobs": 0,
        **chrome_ctx,
    }
    # `{**base_ctx, **tab_ctx}`: tab_ctx wins on key collisions. The
    # `processing` builder deliberately re-supplies `films`, overriding the
    # base value here; that override is intended, not a bug.
    if active_tab == "home":
        tab_ctx = build_home_context(cfg)
    elif active_tab == "search":
        tab_ctx = search.build_search_context(current_slug)
        # Mojica Task 10: ``?q=<text>`` survives push-url navigation back
        # to ``/search`` (HTMX rewrites the bar on every form submit), so
        # the rewritten template can restore the query input value on a
        # full-page reload. The actual results list is not re-fetched
        # here — only the input value is preserved; the HTMX form fires
        # the real /api/search call on submit / keyup.
        q = (request.query_params.get("q") or "").strip()
        if q:
            tab_ctx["query"] = q
        # Mojica Task 13: when the URL carries ``?scene=<id>&film=<slug>``
        # (a timeline-segment link or a deep-share URL into a specific
        # scene), populate the bottom-timeline (``.b-tl``) context. The
        # builder also returns ``selected_film`` (augmented with timeline
        # attrs) + ``selected_scene``, which the right-pane inspector
        # partial picks up on the same render so the .b-rp is visible
        # without an extra HTMX swap. The builder returns ``None`` when
        # the (slug, scene_id) pair cannot be resolved or the film has
        # no keyframe metadata — in which case the timeline partial's
        # self-guard simply renders nothing.
        scene_param = request.query_params.get("scene")
        film_param = request.query_params.get("film")
        if scene_param is not None and film_param:
            try:
                scene_id = int(scene_param)
            except ValueError:
                scene_id = None
            if scene_id is not None:
                timeline_ctx = build_timeline_context(
                    cfg, slug=film_param, scene_id=scene_id, query=q
                )
                if timeline_ctx is not None:
                    tab_ctx.update(timeline_ctx)
    elif active_tab == "rimas":
        # Mojica Task 21: ``?anchor=<slug>/<scene_id>`` is a deep-share URL
        # into a specific anchor scene. Task 22 adds ``?echo=<slug>/<scene_id>``
        # to pre-populate the right-pane inspector with one of the echo
        # cards highlighted. The service handles parsing + falling back
        # to the default anchor / no-echo when the params are absent or
        # malformed, so an unrecognised value never crashes the page.
        anchor_param = request.query_params.get("anchor")
        echo_param = request.query_params.get("echo")
        tab_ctx = build_rimas_context(cfg, anchor=anchor_param, echo=echo_param)
    elif active_tab == "scenes":
        # ``?film=<slug>`` narrows the .c-cp grid to a single film's
        # group; ``slug=None`` keeps the library-wide aggregate view.
        # Without this branch the full-page route rendered the aggregate
        # grid even when the URL bar / sidebar advertised a selected
        # film, so picking a film and the LeftPane marking the row
        # active had no effect on the visible thumbnails. The HTMX
        # fragment routes (``/tab/scenes``, ``/api/scenes``) always
        # threaded the slug through; this branch restores parity.
        # ``?scene=<id>`` deep-link parsing stays a fragment-only
        # concern (the right-pane inspector lives on a separate swap),
        # so ``selected_scene_id`` is left at the builder's default.
        tab_ctx = build_cenas_context(
            cfg,
            slug=current_slug,
            bucket=request.query_params.get("bucket"),
        )
    elif active_tab == "annotate":
        fctx = (
            FilmContext.for_film(cfg, current_slug)
            if current_slug
            else FilmContext.from_config(cfg)
        )
        filter_param = request.query_params.get("filter") or "no_llm"
        scene_param = request.query_params.get("id")
        try:
            scene_id = int(scene_param) if scene_param is not None else None
        except ValueError:
            scene_id = None
        tab_ctx = {
            **build_annotate_context(fctx, filter_param, scene_id),
            "annotate_tab": normalize_annotate_tab(request.query_params.get("tab") or "comments"),
        }
    else:
        tab_ctx = _TAB_CONTEXT_BUILDERS[active_tab]()
    # Mojica chrome kwargs (active_tab=PT slug, compact_lp, has_right_pane) are
    # merged via make_ctx defaults; the EN tab key in base_ctx["active_tab"] is
    # the legacy value still consumed by the in-page tab-bar partial and is
    # intentionally overwritten by _TAB_CHROME[active_tab]["active_tab"] (PT
    # slug) for the new ``data-active-tab`` body attribute. The legacy partial
    # reads ``legacy_active_tab`` instead — set below.
    chrome = _TAB_CHROME.get(active_tab, {})
    merged = {**base_ctx, **tab_ctx}
    # Preserve the EN tab key for the legacy tab-bar / partial dispatch in
    # base.html (selects which partial to include and which tab is marked
    # tab--active). The new ``active_tab`` overlay in the chrome dict carries
    # the PT slug used by the chrome shell.
    merged["legacy_active_tab"] = active_tab
    merged["cfg"] = cfg
    # NOTE: ``current_slug`` is already in ``merged`` via the chrome bag,
    # so it is NOT passed as an explicit kwarg here (would trigger a
    # multiple-values TypeError at call time). The chrome bag is the
    # canonical source for it now.
    return templates.TemplateResponse(
        request,
        "base.html",
        make_ctx(request, **{**merged, **chrome}),
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return render_page(request, "home")


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


@app.get("/rimas", response_class=HTMLResponse)
async def page_rimas(request: Request) -> HTMLResponse:
    """Rimas Visuais (cross-film visual rhymes) full-page route."""
    return render_page(request, "rimas")
