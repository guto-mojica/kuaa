"""A4: map cinemateca.errors -> HTTP envelope (JSON) or HTMX error partial.

The HTTP status for each exception subclass is determined by the F2
single source of truth: :func:`cinemateca.errors.http_status_for`.
This module intentionally does NOT duplicate the status table — it
delegates to ``http_status_for`` so the mapping stays canonical in
``cinemateca/errors.py``.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from api.templates import templates
from cinemateca.errors import CinematecaError, http_status_for


def _is_htmx(request: Request) -> bool:
    return request.headers.get("HX-Request", "").lower() == "true"


def install_error_handlers(app: FastAPI) -> None:
    """Register the CinematecaError exception handler on *app*.

    JSON path: returns the ``{error, code, details, status}`` envelope
    (matching the :class:`api.schemas.ErrorEnvelope` shape).

    HTMX path (``HX-Request: true``): renders
    ``web/templates/partials/error.html`` with the same status code so
    htmx can swap the fragment and swap it in without a JS-level check.
    """

    @app.exception_handler(CinematecaError)
    async def _handle_cinemateca_error(
        request: Request, exc: CinematecaError
    ) -> HTMLResponse | JSONResponse:
        status = http_status_for(exc)
        if _is_htmx(request):
            html = templates.env.get_template("partials/error.html").render(
                code=exc.code,
                error=str(exc),
                status=status,
            )
            return HTMLResponse(html, status_code=status)
        return JSONResponse(
            {
                "error": str(exc),
                "code": exc.code,
                "details": None,
                "status": status,
            },
            status_code=status,
        )
