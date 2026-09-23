"""Unit tests for kuaa.search.hybrid — dispatch + RRF fusion.

These tests target the relocated dispatcher directly via its new home
in ``kuaa.search.hybrid`` (T10). They are deliberately thin —
the 12 service-layer tests in ``tests/test_search_hybrid_service.py``
already cover every retriever_mode + tag-filter + min_similarity
permutation; this file's job is to pin the new import path and prove
the three retriever modes still produce sane top-K orderings from the
relocated module. Behaviour-rich coverage stays in
``test_search_hybrid_service.py``; renaming-equivalent regressions
land here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kuaa.retrieval.bm25 import BM25Index
from kuaa.search.cache import IndexStatus, SearchIndex
from kuaa.search.hybrid import search_hybrid


class _FakeEmbedder:
    """Deterministic stub: every text embeds to ``[1, 0, 0, 0]``.

    Mirrors the contract OpenClipEmbedder satisfies (``encode_text``
    returns a 1-D ``np.ndarray``). Returning a unit vector keeps the
    CLIP path cosine-similar to row 0 of the fixture embeddings.
    """

    def encode_text(self, query: str) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


@pytest.fixture
def small_index(tmp_path: Path) -> SearchIndex:
    """3-scene synthetic SearchIndex with one-hot embeddings.

    Cosine to ``[1, 0, 0, 0]``:
      scene 1 → 1.0   (perfect match)
      scene 2 → 0.5   (mid)
      scene 3 → 0.0   (orthogonal)

    The L2-norm of scene 2's vector is sqrt(0.5) so the post-normalise
    cosine is ~0.707; for ranking purposes only the order matters and
    the order is 1 > 2 > 3 stable across the dispatcher's paths.
    """
    emb = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],  # scene 1
            [0.5, 0.5, 0.0, 0.0],  # scene 2
            [0.0, 1.0, 0.0, 0.0],  # scene 3
        ],
        dtype=np.float32,
    )
    # L2-normalise the rows so the cosine math is exact (matches what
    # the embeddings loader does at index-build time).
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    kf_df = pd.DataFrame(
        [
            {"scene_id": 1, "keyframe_idx": 0, "filepath": str(tmp_path / "1.jpg")},
            {"scene_id": 2, "keyframe_idx": 0, "filepath": str(tmp_path / "2.jpg")},
            {"scene_id": 3, "keyframe_idx": 0, "filepath": str(tmp_path / "3.jpg")},
        ]
    )
    return SearchIndex(status=IndexStatus.OK, embeddings=emb, kf_df=kf_df, embedder=_FakeEmbedder())


@pytest.fixture
def small_bm25() -> BM25Index:
    """BM25 index over 3 descriptions; ``horse`` ↔ scenes 1 & 3; ``house`` ↔ scene 3."""
    return BM25Index.build(
        descriptions=[
            {"scene_id": 1, "description": "a man on a horse"},
            {"scene_id": 2, "description": "interior shot"},
            {"scene_id": 3, "description": "horse and house"},
        ],
        tag_index={},
    )


def test_hybrid_mode_clip_matches_search_text(small_index, small_bm25):
    """``retriever_mode='clip'`` returns the cosine-ranked top-K unchanged.

    The fixture's fake embedder makes scene 1 the unique top-1; scene 2 ranks
    second by cosine; scene 3 is orthogonal. ``min_similarity=0.0`` keeps all
    three in scope so top_k=2 must return [1, 2].
    """
    df = search_hybrid(
        small_index,
        bm25=small_bm25,  # ignored in clip mode
        query="horse",
        tags=[],
        tag_index={},
        top_k=2,
        min_similarity=0.0,
        retriever_mode="clip",
        sem_w=0.7,
        bm25_w=0.3,
    )
    assert list(df["scene_id"]) == [1, 2]


def test_hybrid_mode_bm25_returns_lexical(small_index, small_bm25):
    """``retriever_mode='bm25'`` ignores CLIP and ranks by BM25 over descriptions.

    ``horse`` appears in scenes 1 and 3. The lexical ranking depends on
    BM25 tunables but BOTH scenes must surface in the top-2. We assert
    on the SET (order is rank.bm25 internal detail) — that's enough to
    pin "BM25 path is actually being used, not CLIP".
    """
    df = search_hybrid(
        small_index,
        bm25=small_bm25,
        query="horse",
        tags=[],
        tag_index={},
        top_k=2,
        min_similarity=0.0,
        retriever_mode="bm25",
        sem_w=0.7,
        bm25_w=0.3,
    )
    sids = set(df["scene_id"].tolist())
    assert sids == {1, 3}


def test_hybrid_mode_hybrid_fuses_lists(small_index, small_bm25):
    """``retriever_mode='hybrid'`` runs both paths and fuses by weighted RRF.

    Scene 1 has the perfect CLIP cosine AND BM25-matches ``horse``, so it
    must rank #1 in the fused result regardless of the sem_w/bm25_w split.
    The remaining slots are populated by whichever path's second-place
    survives the RRF blend — we only pin scene 1 at the top, because
    that's the assertion that survives any reasonable weight tweak and
    catches a wholesale regression to one-path-only.
    """
    df = search_hybrid(
        small_index,
        bm25=small_bm25,
        query="horse",
        tags=[],
        tag_index={},
        top_k=3,
        min_similarity=0.0,
        retriever_mode="hybrid",
        sem_w=0.7,
        bm25_w=0.3,
    )
    assert list(df["scene_id"])[0] == 1


@pytest.fixture
def four_scene_index(tmp_path: Path) -> SearchIndex:
    """4 scenes; scenes 3 and 4 are orthogonal to the query vector.

    Wider than ``small_index`` on purpose: the tag-order defect needs more
    candidates than ``top_k`` so the cut has something to throw away.
    """
    emb = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    kf_df = pd.DataFrame(
        [
            {"scene_id": i, "keyframe_idx": 0, "filepath": str(tmp_path / f"{i}.jpg")}
            for i in (1, 2, 3, 4)
        ]
    )
    return SearchIndex(status=IndexStatus.OK, embeddings=emb, kf_df=kf_df, embedder=_FakeEmbedder())


@pytest.fixture
def four_scene_bm25() -> BM25Index:
    """``horse`` matches scenes 1, 2 and 4 with decreasing strength."""
    return BM25Index.build(
        descriptions=[
            {"scene_id": 1, "description": "horse horse horse"},
            {"scene_id": 2, "description": "horse horse"},
            {"scene_id": 3, "description": "house"},
            {"scene_id": 4, "description": "a horse"},
        ],
        tag_index={},
    )


def test_tag_filtered_hybrid_returns_rows_when_tagged_scenes_exist(
    four_scene_index, four_scene_bm25
):
    """The tag filter must run BEFORE fusion, on every leg.

    The metadata leg carries the majority fusion share and was never
    tag-filtered at all, so it put untagged scenes 1 and 2 at the head of the
    fused list. The ``top_k`` cut then kept exactly those two, and the filter
    ran afterwards — deleting both and returning **nothing**, while the two
    tagged scenes that should have been the whole answer sat in the discarded
    tail. Filtering in front of fusion returns them.
    """
    kwargs = dict(
        bm25=four_scene_bm25,
        query="horse",
        tags=["exterior"],
        tag_index={"exterior": [3, 4]},
        top_k=2,
        min_similarity=0.0,
        retriever_mode="hybrid",
        sem_w=0.7,
        bm25_w=0.3,
        metadata_ranked=[(1, 9.0), (2, 8.0)],
        metadata_w=0.65,
    )
    df = search_hybrid(four_scene_index, **kwargs)
    assert set(df["scene_id"]) == {3, 4}, "both tagged scenes must survive the cut"
    assert len(df) == 2, "top_k rows are available and must be returned"


def test_tag_filtered_hybrid_still_empties_on_no_match(small_index, small_bm25):
    """Tags requested + nothing matches → empty, never silently unfiltered.

    Moving the filter in front of fusion must not weaken the AND-intersection
    contract shared with the CLIP and aggregate paths.
    """
    df = search_hybrid(
        small_index,
        bm25=small_bm25,
        query="horse",
        tags=["nonexistent"],
        tag_index={"exterior": [2, 3]},
        top_k=3,
        min_similarity=0.0,
        retriever_mode="hybrid",
        sem_w=0.7,
        bm25_w=0.3,
    )
    assert df.empty
