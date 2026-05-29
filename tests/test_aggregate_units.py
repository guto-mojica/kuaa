"""C1 — per-unit tests for the decomposed aggregate pipeline."""

from __future__ import annotations

from cinemateca.search._aggregate.fusion import fuse_global_rrf


def test_fuse_global_rrf_weights_and_ranks() -> None:
    clip = [(("a", 1), 0.9), (("a", 2), 0.5)]
    bm25 = [(("a", 2), 3.0), (("b", 1), 1.0)]
    fused = fuse_global_rrf([(clip, 0.7), (bm25, 0.3)], k_rrf=60)
    keys = [k for k, _ in fused]
    # scene ("a",2) appears in both lists → highest fused score, ranks first.
    assert keys[0] == ("a", 2)
    assert all(score > 0 for _, score in fused)


def test_fuse_global_rrf_skips_zero_weight_lists() -> None:
    a = [(("a", 1), 1.0)]
    b = [(("b", 1), 1.0)]
    fused = fuse_global_rrf([(a, 1.0), (b, 0.0)], k_rrf=60)
    assert [k for k, _ in fused] == [("a", 1)]


def test_metadata_scorer_scores_exact_tag_and_object_matches() -> None:
    from cinemateca.search._aggregate.scorers import MetadataScorer

    scorer = MetadataScorer()
    scores = scorer.score(
        query="dog",
        descriptions=[{"scene_id": 1, "description": "a dog runs", "objects": ["dog"]}],
        tag_index={"dog": [1]},
        visual_rows=[
            {
                "scene_id": 1,
                "object_detection": {
                    "objects": [{"class": "dog"}],
                    "class_counts": {"dog": 2},
                },
            }
        ],
    )
    assert scores.get(1, 0.0) > 0.0


def test_metadata_scorer_ignores_long_queries() -> None:
    from cinemateca.search._aggregate.scorers import MetadataScorer

    scorer = MetadataScorer()
    # >4 tokens → lexical signal disabled (matches legacy guard).
    assert scorer.score(query="a b c d e", descriptions=[], tag_index={}, visual_rows=[]) == {}
