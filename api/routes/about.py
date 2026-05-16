"""About modal/page and locale-switching routes."""
from __future__ import annotations

from urllib.parse import urlsplit

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from api.deps import make_ctx
from api.templates import templates

router = APIRouter()

# Routes a returned-to path is allowed to be after a locale switch.
# Restricting to known in-app pages (rather than echoing an arbitrary
# same-origin path) keeps the redirect target a closed set even if the
# sanitizer ever regresses.
_SAFE_RETURN_PATHS = {
    "/",
    "/search",
    "/scenes",
    "/annotate",
    "/processing",
    "/about",
}


def _safe_return_path(referer: str | None, host: str | None) -> str:
    """Sanitize a ``Referer`` into a safe in-app return path.

    Same-origin + closed-set policy (open-redirect hardening):

    * an *absolute* URL is trusted ONLY if its netloc equals the
      request's own ``Host`` (so ``http://testserver/scenes`` from the
      real browser is honoured, while ``http://evil.test/scenes`` —
      same path, foreign host — is rejected, as is a non-http scheme
      like ``ftp://``/``javascript:``);
    * a *relative* reference is trusted only if it is a single-``/``
      root-anchored path (rejecting ``//host``, backslashes, control
      chars / CRLF injection);
    * the resulting path must finally be one of the app's own pages
      (:data:`_SAFE_RETURN_PATHS`). Anything else falls back to ``/``.
    """
    if not referer:
        return "/"
    try:
        parts = urlsplit(referer)
    except ValueError:
        return "/"
    if parts.scheme or parts.netloc:
        # Absolute URL: only http/https AND only same host is trusted.
        if parts.scheme not in ("http", "https"):
            return "/"
        if not host or parts.netloc != host:
            return "/"
        path = parts.path
    else:
        path = parts.path
    if not path or not path.startswith("/") or path.startswith("//"):
        return "/"
    if "\\" in path or any(ord(c) < 0x20 for c in path):
        return "/"
    # urlsplit already separated query/fragment; the closed-set
    # membership check below is the final gate.
    return path if path in _SAFE_RETURN_PATHS else "/"


@router.get("/api/about", response_class=HTMLResponse)
async def api_about(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "partials/about_modal.html",
        make_ctx(request, version="0.3.0"),
    )


@router.get("/about", response_class=HTMLResponse)
async def page_about(request: Request) -> HTMLResponse:
    """Full-page About — progressive-enhancement fallback for the modal.

    JS-off users follow the sidebar's real ``href="/about"`` here and
    get the same credits content the modal shows: both this page and
    the modal ``{% include %}`` the shared
    ``partials/_about_credits.html`` partial, so they are
    content-identical by construction. This template just wraps that
    partial so the page stands alone.
    """
    return templates.TemplateResponse(
        request,
        "about_page.html",
        make_ctx(request, version="0.3.0"),
    )


@router.get("/api/locale/{code}")
async def api_set_locale(request: Request, code: str) -> Response:
    """Set the locale cookie and return the user to the page they were on.

    The redirect target is the sanitized ``Referer`` (open-redirect
    safe — see :func:`_safe_return_path`), so switching language no
    longer resets the active tab/page back to ``/``. Works for both
    HTMX (``HX-Redirect``) and a plain JS-off navigation (303 to the
    same path).
    """
    supported = {"pt_BR", "en"}
    locale = code if code in supported else "pt_BR"
    target = _safe_return_path(
        request.headers.get("referer"), request.headers.get("host")
    )

    if request.headers.get("hx-request") == "true":
        # HTMX path: 200 + HX-Redirect so htmx does a full client-side
        # navigation back to the same page (locale cookie applied).
        resp: Response = Response(status_code=200)
        resp.headers["HX-Redirect"] = target
    else:
        # JS-off path: a real 303 navigation so the <a href> works
        # without htmx. 303 forces a GET on the target.
        resp = RedirectResponse(url=target, status_code=303)

    resp.set_cookie("locale", locale, max_age=60 * 60 * 24 * 365, samesite="lax")
    return resp
