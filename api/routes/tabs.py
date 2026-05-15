"""Stub tab routes for tabs that don't yet have a dedicated route module."""
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from api.templates import templates

router = APIRouter()

# /tab/search     → api/routes/search.py
# /tab/scenes     → api/routes/scenes.py
# /tab/annotate   → api/routes/annotate.py


@router.get("/tab/processing", response_class=HTMLResponse)
async def tab_processing(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("partials/processing.html", {"request": request})
