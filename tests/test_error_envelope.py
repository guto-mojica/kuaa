"""A4: uniform CinematecaError->HTTP envelope + status-code fixes.

Tests cover:
  - The http_status_for mapping table (uses the F2 SoT, not the plan's 422
    for RetrievalError — the SoT maps it to 500)
  - Image-upload rejection → 400 (was 200)
  - /api/library/select/{unknown} → 404 (was 200)
  - film_slug_query silent-aggregate fallback is preserved (never 5xx)
  - HTMX requests get an HTML partial, not JSON
"""

from __future__ import annotations

import pytest
from fastapi import APIRouter

from cinemateca.errors import (
    ConfigError,
    IndexMissing,
    RetrievalError,
    UserInputError,
)

# ---------------------------------------------------------------------------
# Mapping table — statuses come from http_status_for (F2 SoT).
# RetrievalError → 500 (NOT 422 as the plan draft said; the SoT wins).
# IndexMissing   → 404 (subclass of RetrievalError, more-specific wins).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc,expected",
    [
        (UserInputError("bad"), 400),
        (IndexMissing("no index"), 404),
        (RetrievalError("retr"), 500),  # SoT says 500, not 422
        (ConfigError("cfg"), 500),
    ],
)
def test_error_maps_to_status_and_envelope(client, exc, expected) -> None:
    """Exception handler converts CinematecaError to a JSON envelope."""
    from api.server import app

    router = APIRouter()
    path = f"/__test_raise/{type(exc).__name__}"

    @router.get(path)
    async def _raise():
        raise exc

    app.include_router(router)
    try:
        r = client.get(path, headers={"accept": "application/json"})
        assert r.status_code == expected, (
            f"{type(exc).__name__}: got {r.status_code}, want {expected}"
        )
        body = r.json()
        assert set(body) >= {"error", "code", "status"}, f"missing envelope keys: {set(body)}"
        assert body["status"] == expected
        assert body["code"]  # F2 base carries a stable .code
    finally:
        app.router.routes[:] = [rt for rt in app.router.routes if getattr(rt, "path", "") != path]


# ---------------------------------------------------------------------------
# Fix 1: image-upload rejection must be 400, not HTML 200
# ---------------------------------------------------------------------------


def test_image_upload_rejection_is_4xx(client) -> None:
    """Oversized / non-image upload must be a client error (400), not HTML 200."""
    files = {"file": ("x.txt", b"not an image", "text/plain")}
    r = client.post("/api/search/image", files=files)
    assert r.status_code == 400, f"got {r.status_code}: {r.text[:200]}"


# ---------------------------------------------------------------------------
# Fix 2: /api/library/select/{unknown} → 404
# ---------------------------------------------------------------------------


def test_library_select_unknown_slug_is_404(client) -> None:
    """/api/library/select/<unknown> must return 404, not 200."""
    r = client.get("/api/library/select/does-not-exist", follow_redirects=False)
    assert r.status_code == 404, f"got {r.status_code}: {r.text[:200]}"


# ---------------------------------------------------------------------------
# Fix 3 (non-breaking contract): film_slug_query silent-aggregate fallback
# ---------------------------------------------------------------------------


def test_film_slug_query_silent_aggregate_is_preserved(client) -> None:
    """Documented contract: an unknown ?film= falls back to aggregate (200), never 5xx.

    Stale cookies and bookmarked URLs with a deleted film slug must NOT
    crash every /tab/* fragment route. Pinned by the behaviour of
    film_slug_query returning None for unknown slugs — see api/deps.py.
    """
    r = client.get("/tab/scenes?film=ghost-slug")
    assert r.status_code == 200, f"got {r.status_code}: {r.text[:200]}"


# ---------------------------------------------------------------------------
# HTMX path: error partial returned instead of JSON
# ---------------------------------------------------------------------------


def test_htmx_request_gets_error_partial_not_json(client) -> None:
    """When HX-Request header is set, handler returns HTML partial, not JSON."""
    from fastapi import APIRouter

    from api.server import app

    router = APIRouter()

    @router.get("/__test_raise_htmx")
    async def _raise():
        raise UserInputError("nope")

    app.include_router(router)
    try:
        r = client.get("/__test_raise_htmx", headers={"HX-Request": "true"})
        assert r.status_code == 400, f"got {r.status_code}"
        assert "text/html" in r.headers["content-type"], (
            f"expected text/html, got {r.headers['content-type']}"
        )
    finally:
        app.router.routes[:] = [
            rt for rt in app.router.routes if getattr(rt, "path", "") != "/__test_raise_htmx"
        ]
