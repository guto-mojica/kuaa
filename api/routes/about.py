"""About modal and locale-switching routes."""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, Response

from api.deps import make_ctx
from api.templates import templates

router = APIRouter()


@router.get("/api/about", response_class=HTMLResponse)
async def api_about(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "partials/about_modal.html",
        make_ctx(request, version="0.3.0"),
    )


@router.get("/api/locale/{code}")
async def api_set_locale(code: str) -> Response:
    """Set the locale cookie and trigger a full-page refresh."""
    supported = {"pt_BR", "en"}
    locale = code if code in supported else "pt_BR"
    resp = Response(status_code=200)
    resp.set_cookie("locale", locale, max_age=60 * 60 * 24 * 365, samesite="lax")
    resp.headers["HX-Refresh"] = "true"
    return resp
