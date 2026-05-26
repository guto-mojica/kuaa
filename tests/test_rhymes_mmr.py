"""Unit tests for MMR rerank over CLIP keyframe embeddings."""

from __future__ import annotations

import numpy as np
import pytest

from cinemateca.rhymes import Rhyme, mmr_rerank


def _unit(v: np.ndarray) -> np.ndarray:
    return (v / (np.linalg.norm(v) or 1.0)).astype("float32")


def test_mmr_lambda_one_equals_pure_relevance():
    """λ=1.0 → MMR is identity (relevance-only). Top of input is top of output."""
    anchor = _unit(np.array([1, 0, 0, 0], dtype="float32"))
    cands = [
        Rhyme(
            film_slug="a",
            scene_id=1,
            score=0.9,
            keyframe_path=None,
            embedding=_unit(np.array([0.9, 0.1, 0, 0])),
        ),
        Rhyme(
            film_slug="a",
            scene_id=2,
            score=0.8,
            keyframe_path=None,
            embedding=_unit(np.array([0.95, 0.05, 0, 0])),
        ),  # near-dup of #1
        Rhyme(
            film_slug="b",
            scene_id=3,
            score=0.6,
            keyframe_path=None,
            embedding=_unit(np.array([0, 0, 1, 0])),
        ),  # diverse
    ]
    out = mmr_rerank(anchor_vec=anchor, candidates=cands, lambda_diversity=1.0, k_final=3)
    # λ=1.0 → relevance ranking unchanged → 1, 2, 3.
    assert [r.scene_id for r in out] == [1, 2, 3]


def test_mmr_lambda_zero_maximises_diversity():
    """λ=0.0 → only the diversity term matters. First pick is most relevant
    (tie-break to relevance for the very first item), subsequent picks
    minimise similarity to already-picked."""
    anchor = _unit(np.array([1, 0, 0, 0], dtype="float32"))
    cands = [
        Rhyme(
            film_slug="a",
            scene_id=1,
            score=0.9,
            keyframe_path=None,
            embedding=_unit(np.array([0.9, 0.1, 0, 0])),
        ),
        Rhyme(
            film_slug="a",
            scene_id=2,
            score=0.8,
            keyframe_path=None,
            embedding=_unit(np.array([0.95, 0.05, 0, 0])),
        ),  # near-dup of #1
        Rhyme(
            film_slug="b",
            scene_id=3,
            score=0.6,
            keyframe_path=None,
            embedding=_unit(np.array([0, 0, 1, 0])),
        ),  # diverse
    ]
    out = mmr_rerank(anchor_vec=anchor, candidates=cands, lambda_diversity=0.0, k_final=3)
    # First pick: scene 1 (highest relevance, no prior picks).
    # Second pick: scene 3 (least similar to 1).
    # Third pick: scene 2 (only one left).
    assert [r.scene_id for r in out] == [1, 3, 2]


def test_mmr_default_lambda_breaks_near_duplicates():
    """λ=0.5 with one cluster of near-duplicates + one diverse outlier
    should put the outlier in the top-3 ahead of the third near-duplicate."""
    anchor = _unit(np.array([1, 0, 0, 0], dtype="float32"))
    cands = [
        # 4 near-duplicates from "film_a"
        Rhyme("a", 1, 0.90, None, _unit(np.array([0.90, 0.10, 0, 0]))),
        Rhyme("a", 2, 0.89, None, _unit(np.array([0.91, 0.09, 0, 0]))),
        Rhyme("a", 3, 0.88, None, _unit(np.array([0.92, 0.08, 0, 0]))),
        Rhyme("a", 4, 0.87, None, _unit(np.array([0.93, 0.07, 0, 0]))),
        # 1 diverse outlier from "film_b"
        Rhyme("b", 5, 0.50, None, _unit(np.array([0.50, 0.5, 0.5, 0.5]))),
    ]
    out = mmr_rerank(anchor_vec=anchor, candidates=cands, lambda_diversity=0.5, k_final=3)
    slugs = [r.film_slug for r in out]
    # MMR with λ=0.5 should pick film_b's outlier inside the top-3,
    # breaking the all-film_a near-duplicate run.
    assert "b" in slugs, f"MMR failed to diversify: got {slugs}"


def test_mmr_truncates_to_k_final():
    anchor = _unit(np.ones(4, dtype="float32"))
    cands = [
        Rhyme("a", i, 0.5, None, _unit(np.random.default_rng(i).standard_normal(4)))
        for i in range(20)
    ]
    out = mmr_rerank(anchor_vec=anchor, candidates=cands, lambda_diversity=0.5, k_final=7)
    assert len(out) == 7


def test_mmr_empty_candidates_returns_empty():
    anchor = _unit(np.ones(4, dtype="float32"))
    assert mmr_rerank(anchor_vec=anchor, candidates=[], lambda_diversity=0.5, k_final=10) == []


def test_mmr_requires_embedding_on_each_rhyme():
    """A Rhyme without an embedding cannot be MMR-reranked — surface this
    explicitly rather than silently returning the input unchanged."""
    anchor = _unit(np.ones(4, dtype="float32"))
    cands = [Rhyme("a", 1, 0.5, None, embedding=None)]
    with pytest.raises(ValueError, match="embedding"):
        mmr_rerank(anchor_vec=anchor, candidates=cands, lambda_diversity=0.5, k_final=1)
