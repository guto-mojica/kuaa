"""Cross-encoder reranker tests.

Fills in the M2 stub at :func:`cinemateca.search.rerank.rerank`.
Tests use the documented ``model='noop'`` escape hatch + an injected
stub model (via ``_load_reranker`` monkeypatch) to avoid HF downloads.
"""

from __future__ import annotations

import sys

# NOTE: ``cinemateca.search.__init__`` re-exports ``rerank`` (the function),
# which shadows the submodule attribute at ``cinemateca.search.rerank``.
# Fetch the module via ``sys.modules`` so ``monkeypatch.setattr`` targets
# the real module object's ``_load_reranker`` symbol.
import cinemateca.search  # noqa: F401  -- ensures the package is imported
from cinemateca.search.rerank import rerank
from cinemateca.search.types import Hit, Query, SearchResult

rerank_mod = sys.modules["cinemateca.search.rerank"]


def _make_result(hits: list[Hit], query_text: str = "anything") -> SearchResult:
    """Build a SearchResult with the M3 typed shape used by ``cinemateca.search``."""
    return SearchResult(
        hits=hits,
        mode="clip",
        weights=None,
        query=Query.text(query_text),
    )


def test_rerank_noop_passes_result_through_unchanged() -> None:
    """``model='noop'`` is the documented passthrough — wider suite depends on it."""
    hits = [
        Hit(scene_id=1, score=0.9, keyframe_path="/p/1.jpg", description="d1"),
        Hit(scene_id=2, score=0.5, keyframe_path="/p/2.jpg", description="d2"),
    ]
    r = _make_result(hits, query_text="anything")
    out = rerank(r, model="noop")
    assert out.hits == hits
    assert out.mode == "clip"


def test_rerank_empty_hits_short_circuits_without_loading_model(monkeypatch) -> None:
    """Empty result must not trigger model load — guards against unnecessary HF download."""
    called = {"loaded": False}

    def _boom(_model_id: str):
        called["loaded"] = True
        raise AssertionError("loader must not be called for an empty hit list")

    monkeypatch.setattr(rerank_mod, "_load_reranker", _boom)
    r = _make_result([], query_text="x")
    out = rerank(r, model="default")
    assert out.hits == []
    assert called["loaded"] is False


def test_rerank_reorders_with_injected_stub_model(monkeypatch) -> None:
    """Verify the rerank logic without loading the real bge cross-encoder."""

    class _Stub:
        def compute_score(self, pairs: list[list[str]]) -> list[float]:
            # Score 10 if doc contains "match", else 0.
            return [10.0 if "match" in d else 0.0 for _q, d in pairs]

    monkeypatch.setattr(rerank_mod, "_load_reranker", lambda _model_id: _Stub())
    hits = [
        Hit(scene_id=1, score=0.9, keyframe_path="/p/1.jpg", description="no signal"),
        Hit(scene_id=2, score=0.5, keyframe_path="/p/2.jpg", description="this is a match"),
        Hit(scene_id=3, score=0.7, keyframe_path="/p/3.jpg", description="nothing"),
    ]
    r = _make_result(hits, query_text="cats")
    out = rerank(r, model="default")

    assert out.hits[0].scene_id == 2  # promoted by the stub
    assert out.hits[0].rerank_score == 10.0
    assert out.hits[1].rerank_score == 0.0
    assert out.hits[2].rerank_score == 0.0
    # original metadata preserved
    assert out.hits[0].keyframe_path == "/p/2.jpg"
    assert out.hits[0].score == 0.5
    # query / mode preserved on the result
    assert out.query.text == "cats"
    assert out.mode == "clip"


def test_rerank_top_k_in_truncates_before_scoring(monkeypatch) -> None:
    """Hits beyond ``top_k_in`` are dropped before the cross-encoder is called."""

    class _Stub:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def compute_score(self, pairs: list[list[str]]) -> list[float]:
            self.calls.append(len(pairs))
            return [1.0] * len(pairs)

    stub = _Stub()
    monkeypatch.setattr(rerank_mod, "_load_reranker", lambda _model_id: stub)
    hits = [
        Hit(scene_id=i, score=1.0 / (i + 1), keyframe_path=f"/p/{i}.jpg", description="x")
        for i in range(30)
    ]
    r = _make_result(hits, query_text="q")
    out = rerank(r, model="default", top_k_in=10)

    assert stub.calls == [10]
    assert len(out.hits) == 10
    # Every returned hit carries a rerank_score (set to 1.0 by the stub)
    assert all(h.rerank_score == 1.0 for h in out.hits)


def test_rerank_custom_model_id_passed_to_loader(monkeypatch) -> None:
    """A non-'default' / non-'noop' value is forwarded as an HF model id."""
    seen: dict[str, str] = {}

    class _Stub:
        def compute_score(self, pairs):
            return [0.0] * len(pairs)

    def _capture(model_id: str):
        seen["model_id"] = model_id
        return _Stub()

    monkeypatch.setattr(rerank_mod, "_load_reranker", _capture)
    hits = [Hit(scene_id=1, score=0.9, keyframe_path="/p/1.jpg", description="d")]
    r = _make_result(hits, query_text="q")
    rerank(r, model="my-org/my-reranker")

    assert seen["model_id"] == "my-org/my-reranker"


def test_rerank_default_resolves_to_bge_v2_m3(monkeypatch) -> None:
    """``model='default'`` resolves to the M3 default cross-encoder id."""
    seen: dict[str, str] = {}

    class _Stub:
        def compute_score(self, pairs):
            return [0.0] * len(pairs)

    def _capture(model_id: str):
        seen["model_id"] = model_id
        return _Stub()

    monkeypatch.setattr(rerank_mod, "_load_reranker", _capture)
    hits = [Hit(scene_id=1, score=0.9, keyframe_path="/p/1.jpg", description="d")]
    r = _make_result(hits, query_text="q")
    rerank(r, model="default")

    assert seen["model_id"] == "BAAI/bge-reranker-v2-m3"
