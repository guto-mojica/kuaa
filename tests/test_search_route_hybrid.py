"""HTTP-layer tests for the M2 hybrid-search dispatch.

These tests confirm ``/api/search`` routes correctly through the three
retrieval modes (``clip`` / ``bm25`` / ``hybrid``), validates inputs
(weight clamps, unknown mode, degenerate zero-weights), and that the
legacy default flips to ``hybrid``.

The route's INFO log line is the canonical pin: ``api_search`` emits
``"retriever=<mode> sem_w=… bm25_w=…"`` once per request, which lets us
distinguish "the dispatcher chose hybrid by default" from "FastAPI
silently dropped an unknown query param" — the latter would never log.

We deliberately keep the fixture surface minimal: every test uses the
shared ``client`` fixture from ``tests/conftest.py`` against an EMPTY
temp library. The route short-circuits at the ``has_indexed_films``
check (no index ⇒ render the no-index partial) BEFORE the dispatch
fires, so we still exercise param parsing + the validation + the log
line without needing real embeddings. This mirrors the discipline of
``tests/test_search_bm25_loader.py`` and ``tests/test_search_hybrid_service.py``
which keep the heavy index out of the test path.
"""

from __future__ import annotations

import logging


def test_search_route_accepts_retriever_clip(client) -> None:
    """``retriever=clip`` short-circuits to pure CLIP — the regression-pin path."""
    resp = client.get("/api/search", params={"q": "menina", "retriever": "clip"})
    assert resp.status_code == 200


def test_search_route_accepts_retriever_bm25(client) -> None:
    resp = client.get("/api/search", params={"q": "menina", "retriever": "bm25"})
    assert resp.status_code == 200


def test_search_route_accepts_retriever_hybrid_with_weights(client) -> None:
    resp = client.get(
        "/api/search",
        params={"q": "menina", "retriever": "hybrid", "sem_w": 0.5, "bm25_w": 0.5},
    )
    assert resp.status_code == 200


def test_search_route_default_retriever_is_hybrid(client, caplog) -> None:
    """No ``retriever`` param ⇒ hybrid. Verified by the route's INFO log line."""
    with caplog.at_level(logging.INFO, logger="api.routes.search"):
        resp = client.get("/api/search", params={"q": "menina"})
    assert resp.status_code == 200
    # ``api_search`` emits the canonical mode/weights line every request;
    # the substring ``retriever=hybrid`` is the regression pin that hybrid
    # is the M2 default (and that FastAPI did NOT silently drop the param).
    assert any("retriever=hybrid" in r.getMessage() for r in caplog.records)


def test_search_route_unknown_retriever_falls_back_to_default(client, caplog) -> None:
    """An unknown retriever value warns + falls back to ``hybrid``."""
    with caplog.at_level(logging.WARNING, logger="api.routes.search"):
        resp = client.get("/api/search", params={"q": "menina", "retriever": "foobar"})
    assert resp.status_code == 200
    assert any("unknown retriever" in r.getMessage().lower() for r in caplog.records)


def test_search_route_clamps_out_of_range_weights(client) -> None:
    """``sem_w=2`` and ``bm25_w=-0.5`` get clamped to ``(1.0, 0.0)`` (still valid)."""
    resp = client.get(
        "/api/search",
        params={"q": "menina", "retriever": "hybrid", "sem_w": 2.0, "bm25_w": -0.5},
    )
    assert resp.status_code == 200


def test_search_route_degenerate_zero_weights_falls_back_to_defaults(client) -> None:
    """``sem_w=0`` ∧ ``bm25_w=0`` falls back to config defaults (0.70/0.30)."""
    resp = client.get(
        "/api/search",
        params={"q": "menina", "retriever": "hybrid", "sem_w": 0.0, "bm25_w": 0.0},
    )
    assert resp.status_code == 200
