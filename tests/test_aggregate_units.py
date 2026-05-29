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


def test_clip_scorer_best_keyframe_per_scene() -> None:
    import numpy as np
    import pandas as pd

    from cinemateca.search._aggregate.scorers import CLIPScorer

    # Two keyframes for scene 1 (rows 0,1), one for scene 2 (row 2).
    embeddings = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)
    kf_df = pd.DataFrame({"scene_id": [1, 1, 2], "filepath": ["a", "b", "c"]})
    text_vec = np.array([1.0, 0.0], dtype=np.float32)
    ranked, best_row_by_sid = CLIPScorer().score(
        embeddings=embeddings,
        kf_df=kf_df,
        text_vec=text_vec,
        min_similarity=0.0,
        allowed_scene_keys=None,
        raw_k=10,
    )
    # scene 1's best row is row 0 (cosine 1.0 > 0.9); ranked desc by cosine.
    assert best_row_by_sid[1] == 0
    assert ranked[0][0] == 1


def test_clip_scorer_applies_min_similarity_floor() -> None:
    import numpy as np
    import pandas as pd

    from cinemateca.search._aggregate.scorers import CLIPScorer

    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    kf_df = pd.DataFrame({"scene_id": [1, 2], "filepath": ["a", "b"]})
    text_vec = np.array([1.0, 0.0], dtype=np.float32)
    ranked, best_row_by_sid = CLIPScorer().score(
        embeddings=embeddings,
        kf_df=kf_df,
        text_vec=text_vec,
        min_similarity=0.5,
        allowed_scene_keys=None,
        raw_k=10,
    )
    # scene 2 (cosine 0.0) is floored out; only scene 1 survives.
    assert [sid for sid, _ in ranked] == [1]
    assert set(best_row_by_sid) == {1}


def test_bm25_scorer_none_index_returns_empty() -> None:
    from cinemateca.search._aggregate.scorers import BM25Scorer

    assert BM25Scorer().score(bm25=None, query="dog", raw_k=10, allowed_scene_keys=None) == []


def test_bm25_scorer_queries_and_filters_by_allowed_keys() -> None:
    from cinemateca.scene_ids import scene_id_key
    from cinemateca.search._aggregate.scorers import BM25Scorer

    class _StubBM25:
        model = object()

        def query(self, q: str, *, top_k: int) -> list[tuple[int, float]]:
            return [(1, 3.0), (2, 1.0)]

    # No filter → both hits pass through verbatim.
    hits = BM25Scorer().score(bm25=_StubBM25(), query="dog", raw_k=10, allowed_scene_keys=None)
    assert hits == [(1, 3.0), (2, 1.0)]
    # Filter to scene 1 only → scene 2 dropped.
    filtered = BM25Scorer().score(
        bm25=_StubBM25(), query="dog", raw_k=10, allowed_scene_keys={scene_id_key(1)}
    )
    assert filtered == [(1, 3.0)]
