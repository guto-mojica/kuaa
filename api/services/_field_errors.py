"""Shared accessible field-error rendering (U1).

One place that renders ``web/templates/partials/_field_error.html`` as a
standalone, out-of-band-swappable fragment for the three input surfaces
(image dropzone, add-film form, search). Keeping it here — not inline in a
route — preserves the deep-modules invariant (routes stay HTTP-shape only)
and means every surface emits byte-identical a11y markup
(``role="alert"`` + ``data-field-error`` + the matching ``id``).

The fragment is rendered through ``make_ctx`` so the per-request ``_``
(gettext) is in scope. Callers pass a STABLE ``message_key`` (not a prose
string): the partial maps the key to a translated literal, which keeps every
UI string inside a template — the Babel catalog is template-only (see
``web/babel.cfg``), so a key->literal map in the partial is the only way both
PT and EN resolve.
"""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import HTMLResponse

from api.deps import make_ctx
from api.templates import templates


def render_field_error_fragment(request: Request, *, slot_id: str, message_key: str) -> str:
    """Render the OOB field-error fragment for *slot_id* with *message_key*.

    ``message_key`` is one of the stable keys handled by
    ``partials/_field_error.html`` (e.g. ``"upload_too_large"``); the partial
    translates it. Returns the rendered HTML string (not a Response) so
    callers can return it at HTTP 200 (search) or fold it into an error
    response (the image upload keeps its honest 4xx — the ``htmx:beforeSwap``
    shim in mojica.js lets the OOB fragment still apply).
    """
    ctx = make_ctx(request, slot_id=slot_id, message_key=message_key, oob=True)
    return templates.env.get_template("partials/_field_error.html").render(**ctx)


def upload_error_response(request: Request, message_key: str) -> HTMLResponse:
    """400 carrying the OOB image-upload field-error fragment (U1).

    Status stays 400 (honest client error, pinned by the A4 envelope test);
    the body is the accessible OOB fragment for ``#image-upload-error``. The
    ``htmx:beforeSwap`` shim in mojica.js permits this fragment to apply
    despite the 4xx (HTMX suppresses body swaps on error codes by default).
    """
    return HTMLResponse(
        render_field_error_fragment(request, slot_id="image-upload-error", message_key=message_key),
        status_code=400,
    )


def prepend_oob(resp: HTMLResponse, oob_fragment: str) -> HTMLResponse:
    """Prepend an OOB fragment to *resp*'s body, preserving status + headers.

    HTMX processes any ``hx-swap-oob`` element anywhere in the response body,
    so prepending the (empty) error-clear fragment to a rendered results body
    both resets a field-error slot and performs the normal target swap in a
    single response.
    """
    body = resp.body.decode(resp.charset) if isinstance(resp.body, bytes) else str(resp.body)
    # Drop content-length: the new (longer) body recomputes it. Keeping the
    # stale value would truncate the response. content-type is re-derived by
    # HTMLResponse, so a leftover header is harmless but dropped for clarity.
    headers = {
        k: v for k, v in resp.headers.items() if k.lower() not in ("content-length", "content-type")
    }
    return HTMLResponse(oob_fragment + body, status_code=resp.status_code, headers=headers)


def submit_triggered(request: Request) -> bool:
    """True only when an HTMX request came from the explicit search SUBMIT.

    The search box fires ``GET /api/search`` on every debounced keystroke
    (``hx-trigger="keyup …"`` on ``#search-input``, trigger id ``search-input``)
    AND on form submit — Enter in the input or the Search button both resolve
    to the form's ``hx-trigger="submit"``, whose ``HX-Trigger`` is the form id
    ``search-text-form``. A validation message must surface ONLY on that
    submit; flashing it mid-typing would be hostile, and a direct / non-HTMX
    hit with no trigger header must stay silent too (the conservative default).
    """
    return request.headers.get("HX-Trigger", "") == "search-text-form"
