"""Tab content routes — each returns an HTMX fragment."""
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from api.templates import templates

router = APIRouter()


@router.get("/tab/search", response_class=HTMLResponse)
async def tab_search(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("partials/search.html", {"request": request})


@router.get("/tab/scenes", response_class=HTMLResponse)
async def tab_scenes(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("partials/scenes.html", {"request": request})


@router.get("/tab/annotate", response_class=HTMLResponse)
async def tab_annotate(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("partials/annotate.html", {"request": request})


@router.get("/tab/processing", response_class=HTMLResponse)
async def tab_processing(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("partials/processing.html", {"request": request})
