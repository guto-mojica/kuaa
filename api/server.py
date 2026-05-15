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

from api.deps import get_config
from api.routes import annotate, scenes, search, tabs
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


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    cfg = get_config()
    from cinemateca.library import scan_library

    films = scan_library(
        raw_dir=Path(cfg.paths.raw_dir),
        metadata_dir=Path(cfg.paths.metadata_dir),
    )
    return templates.TemplateResponse(
        "base.html",
        {
            "request": request,
            "active_tab": "search",
            "processing_jobs": 0,
            "films": films,
            "selected_slug": films[0].slug if films else None,
        },
    )
