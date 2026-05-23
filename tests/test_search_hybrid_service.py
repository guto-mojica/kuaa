"""Service-layer test for ``search_hybrid()``.

Given a CLIP ``SearchIndex`` + a ``BM25Index``, the dispatcher
returns a deduped top-K DataFrame in the same shape ``search_text``
produces. Covers the three retriever modes (``clip``, ``bm25``,
``hybrid``) plus the graceful fallback when ``bm25`` is ``None``.

Fixture strategy: real ``SearchIndex`` (the dataclass is cheap to
construct in-memory — see ``tests/test_search_service.py::TestMinSimilarityFloor``
for the established pattern). The embedder is a tiny stub whose
``encode_text`` returns the first row's vector, so CLIP search
deterministically ranks scene_id=0 first.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from api.services import search as search_service
from cinemateca.retrieval.bm25 import BM25Index

# ───────────────────────────────────────────────────────────────────────
# Fixture helper: hand-built SearchIndex for unit tests.
#
# Mirrors the pattern used by tests/test_search_service.py:
#   - status=IndexStatus.OK
#   - embeddings: float32 L2-normalised (N, D) matrix
#   - kf_df: pandas DataFrame with at least scene_id + a filepath col
#   - embedder: stub with encode_text(query) -> unit vector
# ───────────────────────────────────────────────────────────────────────


def _make_fixture_search_index(tmp_path: Path, n_scenes: int = 3):
    """Build a minimal in-memory ``SearchIndex`` over ``n_scenes`` scenes.

    Embeddings are the first ``n_scenes`` 2-D one-hot vectors (so they
    are trivially L2-normalised). The stub embedder returns
    ``[1.0, 0.0]`` for every query, which makes scene_id=0 the unique
    CLIP top-1 — useful for asserting deterministic ordering in the
    hybrid test below.
    """
    from api.services.search import IndexStatus, SearchIndex

    # 2-D one-hot rows: row i has a 1 in position i % 2; pad with zeros.
    # For n_scenes=3 the rows are [1,0], [0,1], [1,0]. Cosine to [1,0]:
    #   row 0 → 1.0   row 1 → 0.0   row 2 → 1.0
    # Ties on rows 0/2 are broken by row order, so search_text returns
    # scene_id=0 ahead of scene_id=2. Sufficient for "0 is in the top-K".
    vectors = []
    for i in range(n_scenes):
        v = [0.0, 0.0]
        v[i % 2] = 1.0
        vectors.append(v)
    arr = np.array(vectors, dtype="float32")
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)
    kf_df = pd.DataFrame(
        [
            {
                "scene_id": i,
                "keyframe_idx": 0,
                "filepath": str(tmp_path / f"s{i}.jpg"),
            }
            for i in range(n_scenes)
        ]
    )

    class _Embedder:
        def encode_text(self, query):
            return np.array([1.0, 0.0], dtype="float32")

    return SearchIndex(
        status=IndexStatus.OK,
        embeddings=arr,
        kf_df=kf_df,
        embedder=_Embedder(),
    )


# ── Tests ──────────────────────────────────────────────────────────────


def test_hybrid_mode_produces_dataframe_with_scene_id_and_similarity(
    tmp_path: Path,
) -> None:
    bm25 = BM25Index.build(
        descriptions=[
            {"scene_id": 0, "description": "menina chorando"},
            {"scene_id": 1, "description": "homem caminhando"},
            {"scene_id": 2, "description": "carro vermelho"},
        ],
        tag_index={},
    )
    clip_index = _make_fixture_search_index(tmp_path, n_scenes=3)

    df = search_service.search_hybrid(
        clip_index,
        bm25=bm25,
        query="menina",
        tags=[],
        tag_index={},
        top_k=3,
        min_similarity=0.0,
        retriever_mode="hybrid",
        sem_w=0.7,
        bm25_w=0.3,
    )
    assert "scene_id" in df.columns
    assert "similarity" in df.columns
    assert len(df) <= 3
    assert 0 in df["scene_id"].tolist()


def test_clip_mode_ignores_bm25(tmp_path: Path) -> None:
    """``retriever_mode='clip'`` must short-circuit BM25 entirely (regression pin)."""
    clip_index = _make_fixture_search_index(tmp_path, n_scenes=3)
    df = search_service.search_hybrid(
        clip_index,
        bm25=None,
        query="menina",
        tags=[],
        tag_index={},
        top_k=3,
        min_similarity=0.0,
        retriever_mode="clip",
        sem_w=1.0,
        bm25_w=0.0,
    )
    expected = search_service.search_text(clip_index, "menina", [], {}, 3, 0.0)
    assert df["scene_id"].tolist() == expected["scene_id"].tolist()


def test_bm25_mode_returns_ranked_dataframe(tmp_path: Path) -> None:
    """``retriever_mode='bm25'`` runs BM25 only, ignores CLIP scores."""
    bm25 = BM25Index.build(
        descriptions=[
            {"scene_id": 0, "description": "menina chorando"},
            {"scene_id": 1, "description": "homem caminhando"},
            {"scene_id": 2, "description": "carro vermelho"},
        ],
        tag_index={},
    )
    clip_index = _make_fixture_search_index(tmp_path, n_scenes=3)
    df = search_service.search_hybrid(
        clip_index,
        bm25=bm25,
        query="menina",
        tags=[],
        tag_index={},
        top_k=3,
        min_similarity=0.0,
        retriever_mode="bm25",
        sem_w=0.0,
        bm25_w=1.0,
    )
    # BM25 only matches scene_id=0 for query "menina".
    assert "scene_id" in df.columns
    assert "similarity" in df.columns
    assert df["scene_id"].tolist() == [0]


def test_hybrid_falls_back_to_clip_when_bm25_is_none(tmp_path: Path) -> None:
    """When ``bm25`` is ``None`` the dispatcher quietly degrades to CLIP-only."""
    clip_index = _make_fixture_search_index(tmp_path, n_scenes=3)
    df = search_service.search_hybrid(
        clip_index,
        bm25=None,
        query="menina",
        tags=[],
        tag_index={},
        top_k=3,
        min_similarity=0.0,
        retriever_mode="hybrid",
        sem_w=0.7,
        bm25_w=0.3,
    )
    expected = search_service.search_text(clip_index, "menina", [], {}, 3, 0.0)
    assert df["scene_id"].tolist() == expected["scene_id"].tolist()


def test_hybrid_falls_back_to_clip_when_bm25_model_is_none(
    tmp_path: Path,
) -> None:
    """Empty BM25 corpus (``model is None``) also degrades gracefully."""
    empty_bm25 = BM25Index.build(descriptions=[], tag_index={})
    assert empty_bm25.model is None  # guard
    clip_index = _make_fixture_search_index(tmp_path, n_scenes=3)
    df = search_service.search_hybrid(
        clip_index,
        bm25=empty_bm25,
        query="menina",
        tags=[],
        tag_index={},
        top_k=3,
        min_similarity=0.0,
        retriever_mode="hybrid",
        sem_w=0.7,
        bm25_w=0.3,
    )
    expected = search_service.search_text(clip_index, "menina", [], {}, 3, 0.0)
    assert df["scene_id"].tolist() == expected["scene_id"].tolist()


def test_bm25_mode_respects_tag_filter(tmp_path: Path) -> None:
    """``tags`` restricts the BM25-only result set to scenes in ``tag_index``.

    Note the corpus uses distinct discriminator tokens (``chorando`` /
    ``sorrindo`` / ``dancando``) — a single shared token across all 3
    docs has zero IDF in ``rank_bm25`` and ``BM25Index.query`` drops
    zero-score hits, leaving nothing to tag-filter.
    """
    bm25 = BM25Index.build(
        descriptions=[
            {"scene_id": 0, "description": "menina chorando"},
            {"scene_id": 1, "description": "homem sorrindo"},
            {"scene_id": 2, "description": "carro dancando"},
        ],
        tag_index={},
    )
    clip_index = _make_fixture_search_index(tmp_path, n_scenes=3)
    df = search_service.search_hybrid(
        clip_index,
        bm25=bm25,
        query="sorrindo dancando",
        tags=["interior"],
        tag_index={"interior": [1]},
        top_k=3,
        min_similarity=0.0,
        retriever_mode="bm25",
        sem_w=0.0,
        bm25_w=1.0,
    )
    # Only scene 1 is tagged 'interior'; the other matches are filtered out.
    assert df["scene_id"].tolist() == [1]
