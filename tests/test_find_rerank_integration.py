"""C5 — find(rerank=True) returns a reranked, typed SearchResult end-to-end."""

from __future__ import annotations

import sys

import numpy as np
import pytest

import cinemateca.search  # noqa: F401 — ensures the package is imported so sys.modules is populated
import cinemateca.search._dispatch as dispatch_mod
from cinemateca.search._dispatch import find
from cinemateca.search.cache import IndexStatus, SearchIndex
from cinemateca.search.types import Query, SearchResult

# NOTE: cinemateca.search.__init__ re-exports `rerank` (the function),
# which shadows the submodule attribute. Fetch the module via sys.modules
# so monkeypatch.setattr targets the real module object's _load_reranker symbol.
rerank_mod = sys.modules["cinemateca.search.rerank"]


@pytest.fixture()
def stub_index(monkeypatch):
    """Patch load_index → a 2-scene OK index; patch the cross-encoder loader."""
    import pandas as pd

    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    kf_df = pd.DataFrame(
        {
            "scene_id": [1, 2],
            "filepath": ["/p/1.jpg", "/p/2.jpg"],
            "description": ["no signal", "this is a match"],
            "similarity": [0.9, 0.1],
        }
    )

    class _Emb:
        def encode_text(self, q: str) -> np.ndarray:
            return np.array([1.0, 0.0], dtype=np.float32)

    idx = SearchIndex(IndexStatus.OK, embeddings=embeddings, kf_df=kf_df, embedder=_Emb())
    monkeypatch.setattr(dispatch_mod, "load_index", lambda *a, **k: idx)

    class _Stub:
        def compute_score(self, pairs):
            return [10.0 if "match" in d else 0.0 for _q, d in pairs]

    monkeypatch.setattr(rerank_mod, "_load_reranker", lambda _m: _Stub())
    return idx


def test_find_rerank_true_returns_typed_reranked_result(stub_index) -> None:
    from types import SimpleNamespace

    film = SimpleNamespace(slug="a", metadata_dir=None, embeddings_dir=None)
    out = find(Query.of_text("cats"), film=film, mode="clip", top_k=5, rerank=True)
    assert isinstance(out, SearchResult)
    assert out.reranker_applied is True
    # the "match" description is promoted to rank 0 by the cross-encoder.
    assert out.hits[0].scene_id == 2


def test_find_rerank_false_is_passthrough(stub_index) -> None:
    from types import SimpleNamespace

    film = SimpleNamespace(slug="a", metadata_dir=None, embeddings_dir=None)
    out = find(Query.of_text("cats"), film=film, mode="clip", top_k=5, rerank=False)
    assert out.reranker_applied is False


def test_find_rerank_loads_descriptions_from_metadata(monkeypatch, tmp_path) -> None:
    """Real CLIP indexes have NO description column (review #3); rerank must
    load captions from scene_descriptions.json instead of scoring empty
    strings. Without the fix the cross-encoder sees [query, ""] and can't
    reorder — which is what confounded the WS-4 rerank ablation.
    """
    import json

    import pandas as pd

    # Index df WITHOUT a `description` column — the real openclip loader shape.
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    kf_df = pd.DataFrame(
        {"scene_id": [1, 2], "filepath": ["/p/1.jpg", "/p/2.jpg"], "similarity": [0.9, 0.1]}
    )

    class _Emb:
        def encode_text(self, q: str) -> np.ndarray:
            return np.array([1.0, 0.0], dtype=np.float32)

    idx = SearchIndex(IndexStatus.OK, embeddings=embeddings, kf_df=kf_df, embedder=_Emb())
    monkeypatch.setattr(dispatch_mod, "load_index", lambda *a, **k: idx)

    class _Stub:
        def compute_score(self, pairs):
            # Empty descriptions would all score 0 (no reorder); the real
            # caption lets "match" win — proving the description was attached.
            return [10.0 if "match" in d else 0.0 for _q, d in pairs]

    monkeypatch.setattr(rerank_mod, "_load_reranker", lambda _m: _Stub())

    meta_dir = tmp_path / "metadata"
    meta_dir.mkdir()
    (meta_dir / "scene_descriptions.json").write_text(
        json.dumps(
            [
                {"scene_id": 1, "description": "no signal"},
                {"scene_id": 2, "description": "this is a match"},
            ]
        )
    )

    from types import SimpleNamespace

    film = SimpleNamespace(slug="a", metadata_dir=meta_dir, embeddings_dir=None)
    out = find(Query.of_text("cats"), film=film, mode="clip", top_k=5, rerank=True)
    assert out.reranker_applied is True
    assert out.hits[0].scene_id == 2  # promoted via the metadata caption
    assert out.hits[0].description == "this is a match"  # caption was attached
