"""Unit tests for cinemateca.search.clip — the CLIP-only search verbs.

These tests pin the post-extraction contract: ``search_text`` and
``search_image`` consume a loaded :class:`SearchIndex` and return a
scene-deduped, top-K cosine-ranked DataFrame. The legacy tests in
``tests/test_search_service.py`` (TestSearchTextMinSimilarity,
TestSceneDedup) still exercise the same surface via the
``api.services.search`` re-export — this file is the targeted unit for
the new home in :mod:`cinemateca.search.clip`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cinemateca.search.cache import IndexStatus, SearchIndex
from cinemateca.search.clip import search_image, search_text


class _FakeEmbedder:
    """Minimal embedder stub — encodes both text and images to the same
    fixed unit vector so ranking is determined purely by the index's
    pre-computed embeddings.
    """

    def encode_text(self, q: str) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def encode_image_single(self, path) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _build_index() -> SearchIndex:
    """3-scene fixture: scene 1 is a perfect match, scene 2 close, scene 3 orthogonal."""
    emb = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],  # scene 1 — cosine 1.000
            [0.9, 0.1, 0.0, 0.0],  # scene 2 — cosine ~0.994 (post-L2)
            [0.0, 1.0, 0.0, 0.0],  # scene 3 — cosine 0.000
        ],
        dtype=np.float32,
    )
    # Normalise so cosine == dot product (matches the real index contract).
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    kf_df = pd.DataFrame(
        [
            {"scene_id": 1, "filepath": "/p/1.jpg"},
            {"scene_id": 2, "filepath": "/p/2.jpg"},
            {"scene_id": 3, "filepath": "/p/3.jpg"},
        ]
    )
    return SearchIndex(
        status=IndexStatus.OK,
        embeddings=emb,
        kf_df=kf_df,
        embedder=_FakeEmbedder(),
    )


def test_search_text_returns_top_k_by_cosine():
    idx = _build_index()
    df = search_text(idx, "query", tags=[], tag_index={}, top_k=2)
    assert list(df["scene_id"]) == [1, 2]


def test_search_text_min_similarity_floors_results():
    idx = _build_index()
    df = search_text(idx, "query", tags=[], tag_index={}, top_k=10, min_similarity=0.95)
    # Only scene 1 (cosine 1.0) clears the 0.95 floor; scene 2 (~0.994 with
    # the fixture above) also clears it after L2 normalisation. We assert
    # scene 1 is first and the orthogonal scene 3 is excluded.
    scene_ids = list(df["scene_id"])
    assert scene_ids[0] == 1
    assert 3 not in scene_ids


def test_search_image_uses_image_embedder():
    idx = _build_index()
    df = search_image(idx, "/p/query.jpg", top_k=2)
    assert list(df["scene_id"])[0] == 1
