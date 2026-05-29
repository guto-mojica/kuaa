"""Unit tests for cinemateca.search.types — pure dataclass / enum behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from cinemateca.search.types import (
    Filters,
    Hit,
    HybridWeights,
    Query,
    SearchResult,
    UploadRejected,
)


def test_query_text_factory():
    q = Query.of_text("man on horse")
    assert q.text == "man on horse"
    assert q.image_path is None
    assert q.image_bytes is None


def test_query_image_factory():
    q = Query.image(Path("/tmp/frame.jpg"))
    assert q.text is None
    assert q.image_path == Path("/tmp/frame.jpg")


def test_query_rejects_multi_modal():
    with pytest.raises(ValueError, match="exactly one"):
        Query(text="x", image_path=Path("/tmp/y.jpg"))


def test_query_rejects_empty():
    with pytest.raises(ValueError, match="exactly one"):
        Query()


def test_hybrid_weights_defaults():
    w = HybridWeights()
    assert w.sem_w == 0.70
    assert w.bm25_w == 0.30
    assert w.rrf_k == 60


def test_filters_default_empty_tags():
    f = Filters()
    assert f.tags == []
    assert f.min_similarity == 0.0


def test_hit_default_film_fields():
    h = Hit(scene_id=1, score=0.9, keyframe_path="/p/frame.jpg")
    assert h.film_slug is None
    assert h.film_title is None
    assert h.timecode == ""
    assert h.description == ""
    assert h.tags == []


def test_search_result_empty_default():
    q = Query.of_text("x")
    r = SearchResult(hits=[], mode="clip", weights=None, query=q)
    assert r.hits == []
    assert r.no_index is False


def test_upload_rejected_is_exception():
    with pytest.raises(UploadRejected):
        raise UploadRejected("too big")
