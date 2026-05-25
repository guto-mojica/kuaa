"""rerank() is a stub in P1 — M2 fills in the body."""
from __future__ import annotations

import pytest

from cinemateca.search.rerank import rerank
from cinemateca.search.types import Hit, Query, SearchResult


def test_rerank_default_model_raises_not_implemented():
    q = Query.text("x")
    r = SearchResult(hits=[], mode="clip", weights=None, query=q)
    with pytest.raises(NotImplementedError, match="M2 cross-encoder"):
        rerank(r)


def test_rerank_identity_passthrough_for_noop_model():
    """A `model='noop'` returns the input untouched — useful test escape hatch."""
    q = Query.text("x")
    hits = [Hit(scene_id=1, score=0.9, keyframe_path="/p/1.jpg")]
    r = SearchResult(hits=hits, mode="clip", weights=None, query=q)
    out = rerank(r, model="noop")
    assert out.hits == hits
    assert out.mode == "clip"
