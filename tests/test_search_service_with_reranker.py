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
        hits=hits, mode="hybrid", weights=None, query=Query.of_text("q"), no_index=False
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


def test_enabled_auto_resolves_to_gpu_probe(fake_cfg, monkeypatch):
    """``enabled: 'auto'`` defers to the GPU probe: on for CUDA/MPS, off for CPU."""
    from api.services import search as svc

    fake_cfg.retrieval.reranker.enabled = "auto"

    monkeypatch.setattr(svc, "_gpu_available", lambda cfg: True)
    enabled, _model, _top = svc._reranker_settings(fake_cfg)
    assert enabled is True

    monkeypatch.setattr(svc, "_gpu_available", lambda cfg: False)
    enabled, _model, _top = svc._reranker_settings(fake_cfg)
    assert enabled is False


def test_enabled_override_beats_auto(fake_cfg, monkeypatch):
    """A request-level ``?reranker_enabled=`` override wins over ``auto``."""
    from api.services import search as svc

    fake_cfg.retrieval.reranker.enabled = "auto"
    monkeypatch.setattr(svc, "_gpu_available", lambda cfg: True)  # auto would be ON
    enabled, _model, _top = svc._reranker_settings(fake_cfg, enabled_override=False)
    assert enabled is False


def test_gpu_available_treats_probe_failure_as_cpu():
    """A cfg the device probe can't read resolves to CPU → reranker off."""
    from api.services import search as svc

    assert svc._gpu_available(SimpleNamespace()) is False


def test_reranker_default_enabled_mirrors_auto(fake_cfg, monkeypatch):
    """``reranker_default_enabled`` (UI seed) follows the same profile logic."""
    from api.services import search as svc

    fake_cfg.retrieval.reranker.enabled = "auto"
    monkeypatch.setattr(svc, "_gpu_available", lambda cfg: True)
    assert svc.reranker_default_enabled(fake_cfg) is True
    monkeypatch.setattr(svc, "_gpu_available", lambda cfg: False)
    assert svc.reranker_default_enabled(fake_cfg) is False


def test_typed_rerank_boundary_orders_enriched_cards(fake_cfg, monkeypatch):
    """Typed rerank boundary feeds descriptions in and re-emits reordered cards.

    C5: replaces the old ``rerank_template_results`` dict round-trip. The card
    list is lifted to a typed ``SearchResult`` (``cards_to_result``), reranked
    on that typed result (``rerank_search_result`` → ``search_rerank``), then
    projected back to template card dicts in result order (``result_to_cards``),
    carrying ``rerank_score``.
    """
    from api.services import search as svc

    def fake_rerank(result, *, model, top_k_in):
        # Descriptions enriched onto the cards must reach the cross-encoder.
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
    cards = [
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

    result, originals = svc.cards_to_result(cards, query="match", mode="hybrid")
    result = svc.rerank_search_result(result, cfg=fake_cfg, enabled=True)
    out = svc.result_to_cards(result, originals)

    assert [r["scene_id"] for r in out] == [352, 351]
    assert out[0]["rerank_score"] == pytest.approx(10.0)
    # Display-only fields the template reads survive the round-trip untouched.
    assert out[0]["similarity"] == pytest.approx(0.4)
    assert out[1]["similarity"] == pytest.approx(0.9)
