"""ASGI middleware: request correlation + access logging (F5).

Generates (or echoes an inbound) ``X-Request-ID`` per request, stashes it
on ``request.state.request_id`` for downstream handlers/SSE (WS-2 A8),
times the request via :func:`cinemateca.timing.timed`, and emits exactly
one structured access-log line on the ``api.access`` logger.
"""
from __future__ import annotations

import logging
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from cinemateca.timing import timed

access_logger = logging.getLogger("api.access")
REQUEST_ID_HEADER = "X-Request-ID"


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Inject ``X-Request-ID`` and log method/path/status/duration."""

    async def dispatch(self, request: Request, call_next) -> Response:
        rid = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
        request.state.request_id = rid
        with timed() as t:
            response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = rid
        access_logger.info(
            "%s %s -> %s %.1fms",
            request.method,
            request.url.path,
            response.status_code,
            t.elapsed_ms,
            extra={
                "request_id": rid,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round(t.elapsed_ms, 1),
            },
        )
        return response
