"""Unit tests for BM25Index.

The index wraps ``rank_bm25.BM25Okapi`` with a ``query(text, top_k) ->
list[(scene_id, score)]`` API that hides the internal doc-index ↔
scene_id mapping. It does not touch disk — callers pass in already-
loaded descriptions and tag_index. Cache + disk I/O live one layer up
in ``api/services/search.py::_get_bm25_index_for_ctx`` (Task C2), so
the retrieval package stays pure.
"""

from __future__ import annotations

from cinemateca.retrieval.bm25 import BM25Index


def test_index_builds_and_returns_top_k() -> None:
    descriptions = [
        {"scene_id": 0, "description": "menina chorando na chuva"},
        {"scene_id": 1, "description": "homem caminhando na rua"},
        {"scene_id": 2, "description": "carro vermelho"},
    ]
    tag_index: dict = {}
    idx = BM25Index.build(descriptions=descriptions, tag_index=tag_index)
    hits = idx.query("menina", top_k=3)
    assert hits[0][0] == 0
    assert hits[0][1] > 0.0


def test_empty_corpus_does_not_crash() -> None:
    idx = BM25Index.build(descriptions=[], tag_index={})
    assert idx.query("anything", top_k=5) == []


def test_query_with_zero_or_negative_top_k_returns_empty() -> None:
    idx = BM25Index.build(
        descriptions=[{"scene_id": 0, "description": "x"}],
        tag_index={},
    )
    assert idx.query("x", top_k=0) == []
    assert idx.query("x", top_k=-1) == []


def test_query_with_empty_token_query_returns_empty() -> None:
    idx = BM25Index.build(
        descriptions=[{"scene_id": 0, "description": "menina"}],
        tag_index={},
    )
    assert idx.query("", top_k=5) == []
    assert idx.query("!!!", top_k=5) == []
