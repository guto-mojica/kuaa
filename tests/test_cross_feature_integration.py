"""C11 — stacked ``find(hybrid) → rerank`` end-to-end against hermetic indexes.

``hybrid_stub`` monkeypatches the dispatch seams so the ``find(hybrid) →
rerank`` portion fuses two real lists and reranks them without loading any
model or touching disk.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

import kuaa.search  # noqa: F401 — ensures the package is imported so sys.modules is populated
import kuaa.search._dispatch as dispatch_mod
from kuaa.search._dispatch import find
from kuaa.search.cache import IndexStatus, SearchIndex
from kuaa.search.types import Query, SearchResult
from tests._snapshot import assert_snapshot

# NOTE: kuaa.search.__init__ re-exports `rerank` (the function),
# which shadows the submodule attribute. Fetch the module via sys.modules
# so monkeypatch.setattr targets the real module object's _load_reranker symbol.
rerank_mod = sys.modules["kuaa.search.rerank"]


@pytest.fixture()
def hybrid_stub(monkeypatch):
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype=np.float32)
    kf_df = pd.DataFrame(
        {
            "scene_id": [1, 2, 3],
            "filepath": ["/p/1.jpg", "/p/2.jpg", "/p/3.jpg"],
            "description": ["a man on a horse", "interior wall 1959", "a dog"],
            "similarity": [0.9, 0.2, 0.5],
        }
    )

    class _Emb:
        def encode_text(self, q: str) -> np.ndarray:
            return np.array([1.0, 0.0], dtype=np.float32)

    idx = SearchIndex(IndexStatus.OK, embeddings=embeddings, kf_df=kf_df, embedder=_Emb())
    monkeypatch.setattr(dispatch_mod, "load_index", lambda *a, **k: idx)

    # Stub BM25 so the hybrid branch fuses two real lists.
    class _BM25:
        model = object()

        def query(self, q, top_k):
            return [(2, 4.0), (1, 1.0)]

    monkeypatch.setattr(dispatch_mod, "_load_bm25_for_mode", lambda *a, **k: _BM25())

    class _Reranker:
        def compute_score(self, pairs):
            return [float(len(d)) for _q, d in pairs]  # deterministic by desc length

    monkeypatch.setattr(rerank_mod, "_load_reranker", lambda _m: _Reranker())
    return idx


def test_find_hybrid_then_rerank_stacked(hybrid_stub) -> None:
    from types import SimpleNamespace

    film = SimpleNamespace(slug="a", metadata_dir=None, embeddings_dir=None)
    out = find(
        Query.of_text("man on a horse"),
        film=film,
        mode="hybrid",
        top_k=3,
        rerank=True,
    )
    assert isinstance(out, SearchResult)
    assert out.reranker_applied is True
    assert out.fusion_used is True
    serializable = [{"scene_id": h.scene_id, "rerank_score": h.rerank_score} for h in out.hits]
    assert_snapshot("stacked_hybrid_rerank", serializable)
