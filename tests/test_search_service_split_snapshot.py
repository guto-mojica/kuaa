"""Byte-identical snapshot gate for the search service decomposition (A1)."""

from __future__ import annotations

from tests._snapshot import assert_snapshot


def test_api_search_no_index_snapshot(client) -> None:
    # Empty-data client: no index → the no-index empty state. Stable, model-free.
    r = client.get("/api/search?q=river&retriever=clip")
    assert r.status_code == 200
    assert_snapshot("search_service/api_search_no_index", r.text)


def test_tab_search_snapshot(client) -> None:
    r = client.get("/tab/search")
    assert r.status_code == 200
    assert_snapshot("search_service/tab_search", r.text)
