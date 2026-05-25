"""Tests for the M2 cross-encoder reranker (cinemateca.search.rerank).

All tests monkeypatch the CrossEncoder so no model is downloaded.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cinemateca.search.rerank import rerank, rerank_dataframe, _load_descriptions
from cinemateca.search.types import Hit, Query, SearchResult


# ── helpers ────────────────────────────────────────────────────────────────────

def _result(*scene_ids: int, scores: list[float] | None = None) -> SearchResult:
    if scores is None:
        scores = [1.0 - i * 0.1 for i in range(len(scene_ids))]
    hits = [Hit(scene_id=sid, score=s, keyframe_path=f"/p/{sid}.jpg") for sid, s in zip(scene_ids, scores)]
    return SearchResult(hits=hits, mode="clip", weights=None, query=Query.text("test query"))


def _fake_film(tmp_path: Path, descriptions: list[dict] | None = None) -> SimpleNamespace:
    meta_dir = tmp_path / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    if descriptions is not None:
        (meta_dir / "scene_descriptions.json").write_text(json.dumps(descriptions))
    return SimpleNamespace(metadata_dir=meta_dir)


def _fake_ce(scores: list[float]):
    """Return a mock CrossEncoder whose predict() returns ``scores``."""
    ce = MagicMock()
    ce.predict.return_value = np.array(scores, dtype=np.float32)
    return ce


# ── rerank() ──────────────────────────────────────────────────────────────────

def test_rerank_noop_passthrough():
    r = _result(1, 2, 3)
    out = rerank(r, film=SimpleNamespace(metadata_dir=Path("/nonexistent")), model="noop")
    assert out.hits == r.hits


def test_rerank_image_query_bypassed(tmp_path):
    """Image queries have no query text — cross-encoder must not run."""
    hits = [Hit(scene_id=1, score=0.9, keyframe_path="/p/1.jpg")]
    r = SearchResult(hits=hits, mode="clip", weights=None, query=Query.image(Path("/img.jpg")))
    film = _fake_film(tmp_path)
    with patch("cinemateca.search.rerank._get_cross_encoder") as mock_ce:
        out = rerank(r, film=film)
    mock_ce.assert_not_called()
    assert out.hits == hits


def test_rerank_empty_hits_bypassed(tmp_path):
    r = SearchResult(hits=[], mode="clip", weights=None, query=Query.text("rain"))
    film = _fake_film(tmp_path)
    with patch("cinemateca.search.rerank._get_cross_encoder") as mock_ce:
        out = rerank(r, film=film)
    mock_ce.assert_not_called()
    assert out.hits == []


def test_rerank_reorders_hits(tmp_path):
    """Cross-encoder reverses the initial ordering → result is flipped."""
    descriptions = [
        {"scene_id": 1, "description": "a horse"},
        {"scene_id": 2, "description": "a man on a horse"},
        {"scene_id": 3, "description": "empty field"},
    ]
    film = _fake_film(tmp_path, descriptions)
    r = _result(1, 2, 3)  # initial scores descending

    # Cross-encoder says scene 3 is best, then 2, then 1
    with patch("cinemateca.search.rerank._get_cross_encoder", return_value=_fake_ce([0.1, 0.5, 0.9])):
        out = rerank(r, film=film, top_k=3)

    assert [h.scene_id for h in out.hits] == [3, 2, 1]


def test_rerank_trims_to_top_k(tmp_path):
    descriptions = [{"scene_id": i, "description": f"scene {i}"} for i in range(1, 6)]
    film = _fake_film(tmp_path, descriptions)
    r = _result(1, 2, 3, 4, 5)

    with patch("cinemateca.search.rerank._get_cross_encoder", return_value=_fake_ce([0.5, 0.4, 0.9, 0.3, 0.8])):
        out = rerank(r, film=film, top_k=3)

    assert len(out.hits) == 3
    assert [h.scene_id for h in out.hits] == [3, 5, 1]


def test_rerank_preserves_metadata(tmp_path):
    """Mode, weights, query, no_index are carried through unchanged."""
    film = _fake_film(tmp_path, [{"scene_id": 1, "description": "x"}])
    r = _result(1)
    with patch("cinemateca.search.rerank._get_cross_encoder", return_value=_fake_ce([0.7])):
        out = rerank(r, film=film)
    assert out.mode == "clip"
    assert out.query == r.query
    assert out.no_index is False


def test_rerank_cross_encoder_failure_returns_original(tmp_path):
    """If the cross-encoder raises, return the original result unchanged."""
    film = _fake_film(tmp_path, [{"scene_id": 1, "description": "x"}])
    r = _result(1, 2)
    bad_ce = MagicMock()
    bad_ce.predict.side_effect = RuntimeError("model error")
    with patch("cinemateca.search.rerank._get_cross_encoder", return_value=bad_ce):
        out = rerank(r, film=film)
    assert out.hits == r.hits


# ── rerank_dataframe() ────────────────────────────────────────────────────────

def test_rerank_dataframe_noop(tmp_path):
    import pandas as pd

    df = pd.DataFrame({"scene_id": [1, 2], "similarity": [0.9, 0.5], "filepath": ["a", "b"]})
    out = rerank_dataframe(df, query="rain", metadata_dir=tmp_path, model="noop")
    assert list(out["scene_id"]) == [1, 2]


def test_rerank_dataframe_reorders(tmp_path):
    import pandas as pd

    (tmp_path / "scene_descriptions.json").write_text(
        json.dumps([{"scene_id": 1, "description": "horse"}, {"scene_id": 2, "description": "rain"}])
    )
    df = pd.DataFrame({"scene_id": [1, 2], "similarity": [0.9, 0.5], "filepath": ["a", "b"]})
    with patch("cinemateca.search.rerank._get_cross_encoder", return_value=_fake_ce([0.2, 0.8])):
        out = rerank_dataframe(df, query="rain scene", metadata_dir=tmp_path, top_k=2)
    assert list(out["scene_id"]) == [2, 1]


def test_rerank_dataframe_empty_returns_unchanged(tmp_path):
    import pandas as pd

    df = pd.DataFrame(columns=["scene_id", "similarity"])
    with patch("cinemateca.search.rerank._get_cross_encoder") as mock_ce:
        out = rerank_dataframe(df, query="rain", metadata_dir=tmp_path)
    mock_ce.assert_not_called()
    assert out.empty


# ── _load_descriptions() ──────────────────────────────────────────────────────

def test_load_descriptions_happy_path(tmp_path):
    (tmp_path / "scene_descriptions.json").write_text(
        json.dumps([{"scene_id": 1, "description": "a horse"}, {"scene_id": 2, "description": ""}])
    )
    d = _load_descriptions(tmp_path)
    assert d == {1: "a horse", 2: ""}


def test_load_descriptions_missing_file(tmp_path):
    assert _load_descriptions(tmp_path) == {}


def test_load_descriptions_malformed_json(tmp_path):
    (tmp_path / "scene_descriptions.json").write_text("{not json}")
    assert _load_descriptions(tmp_path) == {}
