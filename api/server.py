"""FastAPI application — mounted by uvicorn via app.py."""
import sys
from pathlib import Path

# Safety net for non-installed dev environments
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from api.deps import get_config
from api.routes import tabs
from api.templates import templates

app = FastAPI(title="Cinemateca AI", version="0.3.0")

_BASE = Path(__file__).parent.parent
app.mount("/static", StaticFiles(directory=str(_BASE / "web" / "static")), name="static")

app.include_router(tabs.router)


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
