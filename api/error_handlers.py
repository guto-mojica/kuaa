"""A4: map kuaa.errors -> HTTP envelope (JSON) or HTMX error partial.

The HTTP status for each exception subclass is determined by the F2
single source of truth: :func:`kuaa.errors.http_status_for`.
This module intentionally does NOT duplicate the status table — it
delegates to ``http_status_for`` so the mapping stays canonical in
``kuaa/errors.py``.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from api.templates import templates
from kuaa.errors import KuaaError, http_status_for


def _is_htmx(request: Request) -> bool:
    return request.headers.get("HX-Request", "").lower() == "true"


def install_error_handlers(app: FastAPI) -> None:
    """Register the KuaaError exception handler on *app*.

    JSON path: returns the ``{error, code, details, status}`` envelope
    (matching the :class:`api.schemas.ErrorEnvelope` shape).

    HTMX path (``HX-Request: true``): renders
    ``web/templates/partials/error.html`` with the same status code so
    htmx can swap the fragment and swap it in without a JS-level check.
    """

    @app.exception_handler(KuaaError)
    async def _handle_kuaa_error(
        request: Request, exc: KuaaError
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
