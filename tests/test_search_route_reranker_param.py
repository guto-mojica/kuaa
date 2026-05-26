"""Smoke: ``/api/search`` accepts ``?reranker_enabled=`` without 422.

M3 pre-flight 3.3 shipped the URL surface ahead of the live wiring.
The visible Buscar UI now hides the Rerank control, but the route
parameter stays accepted and logged for back-compat. Live reranking
lands after the production dispatchers migrate from ``DataFrame`` /
``list[dict]`` to ``SearchResult``.

This regression pin keeps the route signature honest: if someone
removes the param or types it wrong, FastAPI would 422 here and older
bookmarks or clients using that query string would break.
"""

from __future__ import annotations

import logging


def test_api_search_accepts_reranker_enabled_param(client) -> None:
    """``?reranker_enabled=true`` must not 422 for compatibility."""
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
    """``?reranker_enabled=false`` is accepted for compatibility too."""
    resp = client.get(
        "/api/search",
        params={"q": "anything", "reranker_enabled": "false"},
    )
    assert resp.status_code in (200, 204)


def test_api_search_reranker_enabled_logged(client, caplog) -> None:
    """Route logs the value so compatibility callers remain observable.

    Until the dispatchers call ``apply_reranker``, the log line is the
    only externally-observable signal that the parameter reached the
    backend. Keep it pinned.
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
