"""The /api/search service dispatcher applies the reranker after retrieval.

Covers Task 3.2 of the M3 pre-flight plan: ``api.services.search.apply_reranker``
reads ``cfg.retrieval.reranker.{enabled,model,top_k_in}`` (with defaults when
the block is absent) and either short-circuits (``enabled=False``) or forwards
to :func:`cinemateca.search.rerank` via the patchable ``search_rerank`` symbol.
No real HF download is triggered — tests use ``model='noop'`` (rerank's
documented passthrough escape hatch) or monkeypatch ``search_rerank``.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest


@pytest.fixture
def fake_cfg():
    """Build a minimal cfg namespace with retrieval.reranker present."""
    return SimpleNamespace(
        retrieval=SimpleNamespace(
            reranker=SimpleNamespace(enabled=True, top_k_in=5, model="noop"),
        ),
    )


def _make_result(n: int = 8):
    """Build a SearchResult with ``n`` synthetic Hits + a text Query."""
    from cinemateca.search.types import Hit, Query, SearchResult

    hits = [
        Hit(scene_id=i, score=1.0 / (i + 1), keyframe_path="", description=f"d{i}")
        for i in range(n)
    ]
    return SearchResult(
        hits=hits, mode="hybrid", weights=None, query=Query.text_query("q"), no_index=False
    )


def test_apply_reranker_passes_through_when_disabled(fake_cfg):
    from api.services import search as svc

    fake_cfg.retrieval.reranker.enabled = False
    r = _make_result()
    out = svc.apply_reranker(r, cfg=fake_cfg)
    assert out is r or out.hits == r.hits  # no reordering, no truncation


def test_apply_reranker_calls_search_rerank_with_top_k_in(fake_cfg, monkeypatch):
    from api.services import search as svc

    captured: dict = {}

    def fake_rerank(result, *, model, top_k_in):
        captured["model"] = model
        captured["top_k_in"] = top_k_in
        return result

    monkeypatch.setattr(svc, "search_rerank", fake_rerank)
    r = _make_result()
    out = svc.apply_reranker(r, cfg=fake_cfg)
    assert captured == {"model": "noop", "top_k_in": 5}
    assert out is r  # fake_rerank passes through unchanged


def test_apply_reranker_with_missing_reranker_block_uses_defaults():
    """A cfg without ``retrieval.reranker`` falls back to in-code defaults
    (enabled=True, model='default', top_k_in=20) without raising. We don't
    trigger the real HF load — use model='noop' for the actual call assertion.
    """
    from api.services import search as svc

    cfg_noop = SimpleNamespace(retrieval=SimpleNamespace(reranker=SimpleNamespace(model="noop")))
    r = _make_result(1)
    out = svc.apply_reranker(r, cfg=cfg_noop)
    assert out.hits[0].scene_id == 0  # noop = passthrough


def test_apply_reranker_with_no_retrieval_attr_uses_defaults(monkeypatch):
    """A cfg with no ``retrieval`` attribute at all is also safe."""
    from api.services import search as svc

    captured: dict = {}

    def fake_rerank(result, *, model, top_k_in):
        captured["model"] = model
        captured["top_k_in"] = top_k_in
        return result

    monkeypatch.setattr(svc, "search_rerank", fake_rerank)
    cfg_empty = SimpleNamespace()
    r = _make_result(1)
    svc.apply_reranker(r, cfg=cfg_empty)
    assert captured == {"model": "default", "top_k_in": 20}


def test_rerank_template_results_orders_enriched_dicts(fake_cfg, monkeypatch):
    """Route-level dict adapter feeds descriptions into the reranker."""
    from api.services import search as svc

    fake_cfg.retrieval.reranker.enabled = False

    def fake_rerank(result, *, model, top_k_in):
        assert [h.description for h in result.hits] == ["less relevant", "exact match"]
        assert model == "noop"
        assert top_k_in == 5
        return replace(
            result,
            hits=[
                replace(result.hits[1], rerank_score=10.0),
                replace(result.hits[0], rerank_score=1.0),
            ],
        )

    monkeypatch.setattr(svc, "search_rerank", fake_rerank)
    rows = [
        {
            "film_slug": "default",
            "scene_id": 351,
            "similarity": 0.9,
            "description": "less relevant",
        },
        {
            "film_slug": "default",
            "scene_id": 352,
            "similarity": 0.4,
            "description": "exact match",
        },
    ]

    out = svc.rerank_template_results(
        rows,
        cfg=fake_cfg,
        query="match",
        mode="hybrid",
        enabled=True,
    )

    assert [r["scene_id"] for r in out] == [352, 351]
    assert out[0]["rerank_score"] == pytest.approx(10.0)
