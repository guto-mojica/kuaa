"""C1 — per-unit tests for the decomposed aggregate pipeline."""

from __future__ import annotations

from kuaa.search._aggregate.fusion import fuse_global_rrf


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
    from kuaa.search._aggregate.scorers import MetadataScorer

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
    from kuaa.search._aggregate.scorers import MetadataScorer

    scorer = MetadataScorer()
    # >4 tokens → lexical signal disabled (matches legacy guard).
    assert scorer.score(query="a b c d e", descriptions=[], tag_index={}, visual_rows=[]) == {}


def test_clip_scorer_best_keyframe_per_scene() -> None:
    import numpy as np
    import pandas as pd

    from kuaa.search._aggregate.scorers import CLIPScorer

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

    from kuaa.search._aggregate.scorers import CLIPScorer

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
    from kuaa.search._aggregate.scorers import BM25Scorer

    assert BM25Scorer().score(bm25=None, query="dog", raw_k=10, allowed_scene_keys=None) == []


def test_bm25_scorer_queries_and_filters_by_allowed_keys() -> None:
    from kuaa.scene_ids import scene_id_key
    from kuaa.search._aggregate.scorers import BM25Scorer

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


def test_film_filter_loads_each_index_once(monkeypatch) -> None:
    """C1 acceptance: the index is loaded ONCE per film, not twice."""
    import kuaa.search._aggregate.film_filter as ff_mod
    from kuaa.search.cache import IndexStatus, SearchIndex

    calls: dict[str, int] = {}

    def _counting_loader(cfg, slug):
        calls[slug] = calls.get(slug, 0) + 1
        return SearchIndex(IndexStatus.OK, embeddings=object(), kf_df=object())

    flt = ff_mod.FilmFilter(load_index=_counting_loader)
    candidates = flt.candidates(cfg=object(), slugs=["a", "b"])
    assert {c.slug for c in candidates} == {"a", "b"}
    assert calls == {"a": 1, "b": 1}  # exactly one load per film


def test_film_filter_skips_non_ok_and_unregistered() -> None:
    """A non-OK index and a ValueError-raising slug are both dropped."""
    from kuaa.search._aggregate.film_filter import FilmFilter
    from kuaa.search.cache import IndexStatus, SearchIndex

    def _loader(cfg, slug):
        if slug == "missing":
            return SearchIndex(IndexStatus.MISSING)
        if slug == "ghost":
            raise ValueError(f"Film not registered: {slug!r}")
        return SearchIndex(IndexStatus.OK, embeddings=object(), kf_df=object())

    candidates = FilmFilter(load_index=_loader).candidates(
        cfg=object(), slugs=["ok", "missing", "ghost"]
    )
    assert [c.slug for c in candidates] == ["ok"]


def test_materialize_hits_clip_and_bm25_fallback_rows() -> None:
    """Best-row when present; first-matching kf_df row as BM25-only fallback."""
    import pandas as pd

    from kuaa.search._aggregate.materialize import materialize_hits

    class _Film:
        title = "A"

    kf_df = pd.DataFrame({"scene_id": [1, 1, 2], "filepath": ["s1a", "s1b", "s2"]})
    per_film = {
        "a": {
            "film": _Film(),
            "kf_df": kf_df,
            "best_row_by_sid": {1: 1},  # scene 1 surfaced via CLIP best row = 1
            "fps": 24.0,
            "meta_by_scene": {1: {"start_time_s": 2.0}, 2: {"start_time_s": 0.0}},
        }
    }
    ranked = [(("a", 1), 0.9), (("a", 2), 0.4)]
    hits = materialize_hits(ranked, per_film, top_k=5)
    assert [h["scene_id"] for h in hits] == [1, 2]
    # scene 1 uses best_row_by_sid[1] → row index 1 → "s1b".
    assert hits[0]["keyframe_path"] == "s1b"
    assert hits[0]["timecode"]  # start_time_s 2.0 > 0 → non-empty SMPTE
    # scene 2 has no best row → first kf_df row for sid 2 → "s2"; start 0 → "".
    assert hits[1]["keyframe_path"] == "s2"
    assert hits[1]["timecode"] == ""
