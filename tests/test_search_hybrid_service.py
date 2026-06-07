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


def test_hybrid_backfills_filepath_for_bm25_only_hits(tmp_path: Path) -> None:
    """A BM25-only hit (scene CLIP did not surface) must come back with a
    populated ``filepath`` column.

    Setup: 3 scenes. Stub embedder returns ``[1, 0]`` → cosine on each
    row is row-vector[0]. Rows: [1,0], [0,1], [1,0]. So scene 1's cosine
    is 0.0 — with ``min_similarity=0.5`` it never enters ``clip_df``.
    BM25 sees only scene 1 (its description carries the unique token
    ``"chorando"``). In hybrid mode the fused list contains scene 1
    from the BM25 side only; the row that gets surfaced must have its
    ``filepath`` backfilled from ``index.kf_df`` — currently the backfill
    branch in ``_fused_to_dataframe`` checks a column name that doesn't
    exist (``img_filename`` instead of ``filepath``), so the backfill
    never fires and the row's filepath is NaN.
    """
    import pandas as pd

    bm25 = BM25Index.build(
        descriptions=[
            {"scene_id": 0, "description": "homem caminhando"},
            {"scene_id": 1, "description": "menina chorando"},
            {"scene_id": 2, "description": "carro vermelho"},
        ],
        tag_index={},
    )
    clip_index = _make_fixture_search_index(tmp_path, n_scenes=3)

    df = search_service.search_hybrid(
        clip_index,
        bm25=bm25,
        query="chorando",
        tags=[],
        tag_index={},
        top_k=3,
        min_similarity=0.5,  # filters scene 1 (cosine 0.0) from CLIP
        retriever_mode="hybrid",
        sem_w=0.7,
        bm25_w=0.3,
    )
    assert 1 in df["scene_id"].tolist(), "BM25-only scene 1 must appear in fused output"
    row = df[df["scene_id"] == 1].iloc[0]
    assert "filepath" in df.columns, "fused DataFrame must carry filepath column"
    assert not pd.isna(row["filepath"]), (
        f"BM25-only hit must have a non-NaN filepath after backfill — got {row['filepath']!r}"
    )
    assert str(row["filepath"]).endswith("s1.jpg"), (
        f"backfill must resolve to the right keyframe; got {row['filepath']!r}"
    )


def test_bm25_with_tags_on_empty_tag_index_returns_empty(tmp_path: Path) -> None:
    """Tags requested but the film has NO tag_index at all → empty result.

    Per-film route loads ``tag_index = load_tag_index(ctx.metadata_dir)
    if tags else {}``. On a tagless film this is ``{}``. The current
    ``_bm25_hits_to_dataframe`` short-circuits the tag filter via
    ``if tags and tag_index:`` and returns BM25 hits UNFILTERED — user
    sees results that have no association with the requested tag.
    The correct behaviour: no scene has the tag → empty result.
    """
    bm25 = BM25Index.build(
        descriptions=[
            {"scene_id": 0, "description": "homem caminhando"},
            {"scene_id": 1, "description": "menina chorando"},
            {"scene_id": 2, "description": "carro vermelho"},
        ],
        tag_index={},
    )
    clip_index = _make_fixture_search_index(tmp_path, n_scenes=3)

    df = search_service.search_hybrid(
        clip_index,
        bm25=bm25,
        query="chorando",
        tags=["nonexistent-tag"],
        tag_index={},  # film has no tags at all
        top_k=3,
        min_similarity=0.0,
        retriever_mode="bm25",
        sem_w=0.0,
        bm25_w=1.0,
    )
    assert df.empty, f"empty tag_index + tag query must return empty; got {df['scene_id'].tolist()}"


def test_hybrid_with_tags_on_empty_tag_index_returns_empty(tmp_path: Path) -> None:
    """Same as the bm25 case but for hybrid mode + _fused_to_dataframe.

    Without this guard, hybrid mode on a tagless film silently ignores
    the tag filter on the BM25-only-backfill path.
    """
    bm25 = BM25Index.build(
        descriptions=[
            {"scene_id": 0, "description": "homem caminhando"},
            {"scene_id": 1, "description": "menina chorando"},
            {"scene_id": 2, "description": "carro vermelho"},
        ],
        tag_index={},
    )
    clip_index = _make_fixture_search_index(tmp_path, n_scenes=3)

    df = search_service.search_hybrid(
        clip_index,
        bm25=bm25,
        query="chorando",
        tags=["nonexistent-tag"],
        tag_index={},
        top_k=3,
        min_similarity=0.0,
        retriever_mode="hybrid",
        sem_w=0.7,
        bm25_w=0.3,
    )
    assert df.empty, f"empty tag_index + tag query must return empty; got {df['scene_id'].tolist()}"


def test_search_hybrid_respects_rrf_k_override(tmp_path: Path) -> None:
    """``search_hybrid`` honours its ``rrf_k`` kwarg.

    RRF score for a rank-r doc is ``weight / (rrf_k + r)``. With rrf_k=1
    the top-1 contribution is ``weight / 2``; with rrf_k=60 it's
    ``weight / 61``. The top-1 fused score with rrf_k=1 must therefore
    be ≥ 5× larger than with rrf_k=60 — same ordering, sharper magnitude.

    This is the unit-level guarantee the config knob
    ``cfg.search.bm25.rrf_k`` relies on (route plumbing comes in a
    separate test).
    """
    bm25 = BM25Index.build(
        descriptions=[
            {"scene_id": 0, "description": "menina chorando"},
            {"scene_id": 1, "description": "homem caminhando"},
            {"scene_id": 2, "description": "carro vermelho"},
        ],
        tag_index={},
    )
    clip_index = _make_fixture_search_index(tmp_path, n_scenes=3)

    common = dict(
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
    df_default = search_service.search_hybrid(clip_index, **common, rrf_k=60)
    df_small = search_service.search_hybrid(clip_index, **common, rrf_k=1)
    # Both run on the same data — top scene_id must be stable.
    assert df_default["scene_id"].tolist()[0] == df_small["scene_id"].tolist()[0]
    # rrf_k=1 yields ~5× larger top-1 score than rrf_k=60.
    top_default = float(df_default["similarity"].iloc[0])
    top_small = float(df_small["similarity"].iloc[0])
    assert top_small > top_default * 5, (
        f"rrf_k must change fused score magnitude: small={top_small} default={top_default}"
    )


def _make_multi_keyframe_index(tmp_path: Path):
    """Build a SearchIndex with 2 keyframes per scene for 2 scenes.

    Geometry vs query [1, 0]:
      scene 0 row 0 [1, 0]      cosine 1.000  (best of scene 0)
      scene 0 row 1 [0.5, 0.5]  cosine 0.707
      scene 1 row 0 [0.1, 0.9]  cosine 0.110
      scene 1 row 1 [0.3, 0.95] cosine 0.301  (best of scene 1)

    With ``min_similarity=0.5`` BOTH keyframes of scene 1 fall below
    the floor — scene 1 is BM25-only in the hybrid path. The keyframe
    surfaced for scene 1 must still be the best-cosine row of that
    scene (row 1, ``s1k1.jpg``), not the first row blindly.
    """
    from api.services.search import IndexStatus, SearchIndex

    vectors = [
        [1.0, 0.0],  # scene 0 row 0
        [0.5, 0.5],  # scene 0 row 1
        [0.1, 0.9],  # scene 1 row 0
        [0.3, 0.95],  # scene 1 row 1
    ]
    arr = np.array(vectors, dtype="float32")
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)
    kf_df = pd.DataFrame(
        [
            {"scene_id": 0, "keyframe_idx": 0, "filepath": str(tmp_path / "s0k0.jpg")},
            {"scene_id": 0, "keyframe_idx": 1, "filepath": str(tmp_path / "s0k1.jpg")},
            {"scene_id": 1, "keyframe_idx": 0, "filepath": str(tmp_path / "s1k0.jpg")},
            {"scene_id": 1, "keyframe_idx": 1, "filepath": str(tmp_path / "s1k1.jpg")},
        ]
    )

    class _Embedder:
        def encode_text(self, query):
            return np.array([1.0, 0.0], dtype="float32")

    return SearchIndex(status=IndexStatus.OK, embeddings=arr, kf_df=kf_df, embedder=_Embedder())


def test_hybrid_picks_best_cosine_keyframe_for_bm25_only_scene(tmp_path: Path) -> None:
    """BM25-only scene with multiple keyframes must surface the best-cosine row.

    Per the docstring of ``_make_multi_keyframe_index``: scene 1 has
    two keyframes (``s1k0.jpg`` cosine 0.110, ``s1k1.jpg`` cosine 0.301)
    both below the ``min_similarity=0.5`` floor; BM25 surfaces the scene.
    The displayed keyframe must be ``s1k1.jpg`` — the best-cosine row
    of that scene — for parity with how CLIP-side hits pick keyframes
    (best-per-scene). The current ``iloc[0]`` fallback would surface
    ``s1k0.jpg``, inconsistent with the rest of the result page.
    """
    import pandas as pd

    bm25 = BM25Index.build(
        descriptions=[
            {"scene_id": 0, "description": "homem caminhando"},
            {"scene_id": 1, "description": "menina chorando"},
            # Pad to ≥3 distinct docs so BM25 idf on 'chorando' is positive.
            {"scene_id": 99, "description": "carro vermelho na estrada"},
        ],
        tag_index={},
    )
    clip_index = _make_multi_keyframe_index(tmp_path)

    df = search_service.search_hybrid(
        clip_index,
        bm25=bm25,
        query="chorando",
        tags=[],
        tag_index={},
        top_k=3,
        min_similarity=0.5,  # scene 1's best cosine 0.301 is still below floor
        retriever_mode="hybrid",
        sem_w=0.7,
        bm25_w=0.3,
    )
    assert 1 in df["scene_id"].tolist(), "BM25-only scene 1 must surface"
    row = df[df["scene_id"] == 1].iloc[0]
    assert not pd.isna(row["filepath"]), "backfill must populate filepath"
    fp = str(row["filepath"])
    assert fp.endswith("s1k1.jpg"), (
        f"BM25-only scene with multi-keyframes must surface the "
        f"best-cosine row (s1k1.jpg); got {fp}"
    )


def test_bm25_mode_picks_best_cosine_keyframe(tmp_path: Path) -> None:
    """Same invariant as the hybrid test, exercised through ``retriever_mode='bm25'``."""

    bm25 = BM25Index.build(
        descriptions=[
            {"scene_id": 0, "description": "homem caminhando"},
            {"scene_id": 1, "description": "menina chorando"},
            {"scene_id": 99, "description": "carro vermelho na estrada"},
        ],
        tag_index={},
    )
    clip_index = _make_multi_keyframe_index(tmp_path)

    df = search_service.search_hybrid(
        clip_index,
        bm25=bm25,
        query="chorando",
        tags=[],
        tag_index={},
        top_k=3,
        min_similarity=0.5,
        retriever_mode="bm25",
        sem_w=0.0,
        bm25_w=1.0,
    )
    assert 1 in df["scene_id"].tolist()
    row = df[df["scene_id"] == 1].iloc[0]
    fp = str(row["filepath"])
    assert fp.endswith("s1k1.jpg"), (
        f"bm25-mode multi-keyframe scene must surface best-cosine row; got {fp}"
    )


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
