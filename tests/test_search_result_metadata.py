"""C9 — SearchResult carries per-query metadata."""

from __future__ import annotations

from kuaa.search.types import Hit, Query, SearchResult


def test_search_result_metadata_defaults() -> None:
    r = SearchResult(hits=[], mode="clip", weights=None, query=Query.of_text("x"))
    assert r.fusion_used is False
    assert r.reranker_applied is False
    assert r.retriever_mode == "clip"
    assert r.num_films_searched == 0
    assert r.latency_ms is None


def test_search_result_metadata_populated() -> None:
    r = SearchResult(
        hits=[Hit(scene_id=1, score=0.5, keyframe_path="/p.jpg")],
        mode="hybrid",
        weights=None,
        query=Query.of_text("x"),
        fusion_used=True,
        reranker_applied=True,
        retriever_mode="hybrid",
        num_films_searched=2,
        latency_ms=12.5,
    )
    assert r.fusion_used and r.reranker_applied
    assert r.num_films_searched == 2
    assert r.latency_ms == 12.5
