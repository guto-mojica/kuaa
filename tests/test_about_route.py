"""Snapshot test for the /api/about endpoint.

Behavior-preserving gate: after the about_service.py decomposition,
the rendered HTML must be byte-identical to what the original service produced.
Record/refresh with ``UPDATE_SNAPSHOTS=1 uv run pytest tests/test_about_route.py -q``.
"""

from __future__ import annotations

from tests._snapshot import assert_snapshot


def test_api_about_snapshot(client) -> None:
    r = client.get("/api/about")
    assert r.status_code == 200
    assert_snapshot("about_service/api_about", r.text)
