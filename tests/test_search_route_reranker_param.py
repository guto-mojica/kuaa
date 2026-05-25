"""Smoke: ``/api/search`` accepts ``?reranker_enabled=`` without 422.

M3 pre-flight 3.3 ships the chip-toggle URL surface ahead of the live
wiring. The route parameter exists; today it is *accepted and logged*
only. Live reranking lands in Task 3.2b, after the production
dispatchers migrate from ``DataFrame`` / ``list[dict]`` to
``SearchResult``.

This regression pin keeps the route signature honest: if someone
removes the param or types it wrong, FastAPI would 422 here and the
UI's hidden mirror would silently fail every search submission.
"""

from __future__ import annotations

import logging


def test_api_search_accepts_reranker_enabled_param(client) -> None:
    """``?reranker_enabled=true`` must not 422 — the chip mirror depends on it."""
    resp = client.get(
        "/api/search",
        params={"q": "anything", "reranker_enabled": "true"},
    )
    # Empty-data client returns the no-index empty state (200, HTML); a
    # short-query early-return would also be 200 with an empty body. The
    # only failure mode this pin cares about is 422 (param rejected /
    # missing from signature).
    assert resp.status_code in (
        200,
        204,
    ), f"Route rejected ?reranker_enabled= with {resp.status_code}: {resp.text[:200]}"


def test_api_search_accepts_reranker_enabled_false(client) -> None:
    """``?reranker_enabled=false`` is the chip's default-off shape — also must not 422."""
    resp = client.get(
        "/api/search",
        params={"q": "anything", "reranker_enabled": "false"},
    )
    assert resp.status_code in (200, 204)


def test_api_search_reranker_enabled_logged(client, caplog) -> None:
    """Route logs the toggle so we can confirm the chip reaches the server.

    Until Task 3.2b wires ``apply_reranker`` into the dispatcher path,
    the log line is the only externally-observable signal that the chip
    is reaching the backend. Keep it pinned.
    """
    with caplog.at_level(logging.INFO, logger="api.routes.search"):
        resp = client.get(
            "/api/search",
            params={"q": "menina", "reranker_enabled": "true"},
        )
    assert resp.status_code in (200, 204)
    assert any(
        "reranker_enabled=True" in r.getMessage() for r in caplog.records
    ), "expected api_search INFO log to echo reranker_enabled=True"
