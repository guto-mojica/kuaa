"""Smoke: ``/api/search`` accepts ``?reranker_enabled=`` without 422.

The Buscar UI sends this request-level toggle for text results. The route
logs the value and applies the reranker after card enrichment, where scene
descriptions are available for the cross-encoder.

This regression pin keeps the route signature honest: if someone
removes the param or types it wrong, FastAPI would 422 here and older
bookmarks or clients using that query string would break.
"""

from __future__ import annotations

import logging


def test_api_search_accepts_reranker_enabled_param(client) -> None:
    """``?reranker_enabled=true`` must not 422."""
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
    """``?reranker_enabled=false`` is accepted too."""
    resp = client.get(
        "/api/search",
        params={"q": "anything", "reranker_enabled": "false"},
    )
    assert resp.status_code in (200, 204)


def test_api_search_reranker_enabled_logged(client, caplog) -> None:
    """Route logs the value so rerank requests remain observable."""
    with caplog.at_level(logging.INFO, logger="api.routes.search"):
        resp = client.get(
            "/api/search",
            params={"q": "menina", "reranker_enabled": "true"},
        )
    assert resp.status_code in (200, 204)
    assert any(
        "reranker_enabled=True" in r.getMessage() for r in caplog.records
    ), "expected api_search INFO log to echo reranker_enabled=True"


def test_api_search_threads_reranker_toggle_into_result_adapter(client, monkeypatch) -> None:
    """The route threads the request toggle into the typed rerank boundary.

    C5: the dict round-trip adapter (``rerank_template_results``) is gone; the
    render layer lifts enriched cards to a typed ``SearchResult``
    (``cards_to_result``, receiving query+mode) and reranks that typed result
    (``rerank_search_result``, receiving the request ``enabled`` toggle).
    """
    import api.routes.search as route

    captured: dict = {}

    monkeypatch.setattr(
        route.search_service,
        "dispatch_text_search",
        lambda *args: ([{"scene_id": 351, "score": 0.7, "film_slug": "default"}], False),
    )
    monkeypatch.setattr(
        route.search_service,
        "aggregate_hits_to_template_dicts",
        lambda cfg, payload: payload,
    )
    monkeypatch.setattr(
        route.search_service,
        "enrich_hits_with_film_metadata",
        lambda cfg, rows, per_film_slug=None: [
            {
                "film_slug": "default",
                "scene_id": 351,
                "similarity": 0.7,
                "description": "a man walking outdoors",
            }
        ],
    )
    monkeypatch.setattr(route.search_service, "films_by_id_lookup", lambda cfg: {})

    real_cards_to_result = route.search_service.cards_to_result

    def spy_cards_to_result(cards, *, query, mode="hybrid"):
        captured.update({"query": query, "mode": mode})
        return real_cards_to_result(cards, query=query, mode=mode)

    def fake_rerank(result, *, cfg, enabled=None):
        captured["enabled"] = enabled
        return result

    monkeypatch.setattr(route.search_service, "cards_to_result", spy_cards_to_result)
    monkeypatch.setattr(route.search_service, "rerank_search_result", fake_rerank)

    resp = client.get(
        "/api/search",
        params={"q": "walking", "reranker_enabled": "true", "retriever": "hybrid"},
    )

    assert resp.status_code == 200
    assert captured == {"query": "walking", "mode": "hybrid", "enabled": True}


def test_first_stage_pool_widens_for_reranker(client, monkeypatch) -> None:
    """1.1: with rerank on, the first stage fetches >= ``top_k_in`` candidates.

    The cross-encoder scores only the top ``top_k_in`` hits it is handed, so the
    first stage must surface that depth — otherwise rerank can only reorder the
    already-visible page and never promote a deeper candidate.
    """
    import api.routes.search as route

    captured: dict = {}

    def capture_dispatch(*args):
        # dispatch_text_search(cfg, ctx, q, tags, top_k, min_sim, retriever, sw, bw, rrf_k)
        captured["first_stage_k"] = args[4]
        return ([], False)

    monkeypatch.setattr(route.search_service, "dispatch_text_search", capture_dispatch)
    resp = client.get(
        "/api/search",
        params={"q": "menina", "reranker_enabled": "true", "top_k": 8},
    )
    assert resp.status_code in (200, 204)
    # top_k_in defaults to 20; top_k=8 < 20, so the pool must widen to >= 20.
    assert captured["first_stage_k"] >= 20


def test_first_stage_pool_covers_offset_when_rerank_off(client, monkeypatch) -> None:
    """1.1: paging is honoured even with rerank off.

    The final ``[offset : offset + top_k]`` slice would index past a
    ``top_k``-length list, so the first stage must fetch ``top_k + offset``.
    """
    import api.routes.search as route

    captured: dict = {}

    def capture_dispatch(*args):
        captured["first_stage_k"] = args[4]
        return ([], False)

    monkeypatch.setattr(route.search_service, "dispatch_text_search", capture_dispatch)
    resp = client.get(
        "/api/search",
        params={"q": "menina", "reranker_enabled": "false", "top_k": 5, "offset": 10},
    )
    assert resp.status_code in (200, 204)
    assert captured["first_stage_k"] == 15  # 5 + 10, no rerank widening
