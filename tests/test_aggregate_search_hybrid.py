"""Cross-film hybrid dispatch in ``aggregate_search``.

Note: ``_get_embedder`` monkeypatches target ``cinemateca.search.aggregate``
(the module), accessed via ``sys.modules`` because the ``aggregate`` function
name shadows the submodule in the ``cinemateca.search`` package namespace.

Task D2 — the per-film fan-out branches on ``retriever_mode``:

  * ``"clip"``   — inline cosine over the per-film CLIP index, identical
    to pre-M2 behaviour (regression pin).
  * ``"bm25"``   — per-film ``BM25Index.query`` results, tag-filtered.
  * ``"hybrid"`` — RRF of the two with ``sem_w`` / ``bm25_w`` weights.

The cross-film merge stage (score-sorted union + dedupe by
``(film_slug, scene_id)`` + top_k cut) is shared across all three modes.

No shared ``two_film_library_cfg`` fixture exists in ``conftest.py``;
the existing pattern (see ``tests/test_multi_film_search.py``) is to
inline-build the per-film directory layout + CLIP index + tag/desc JSON
inside each test using ``register_film`` and the helpers replicated
below. We follow that pattern verbatim so this file stays
self-contained and skips a global-fixture refactor that would touch
unrelated tests.

The CLIP text-encoder is stubbed (real OpenClipEmbedder load is ~4 s);
``_get_bm25_index_for_ctx`` itself is NOT stubbed — we exercise the
real loader against on-disk descriptions / tags so the BM25 + RRF
paths are integration-tested end-to-end.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import cinemateca.search.aggregate as _csa_module_ref  # noqa: F401 — ensure loaded
from api.services.search import aggregate_search
from cinemateca.library import register_film

# The submodule is shadowed in cinemateca.search by the `aggregate` function
# re-export; access via sys.modules to reach the module object reliably.
_AGGREGATE_MODULE = sys.modules["cinemateca.search.aggregate"]

# ── Inline two-film fixture helpers (mirrors test_multi_film_search.py) ──────


def _make_film_with_embeddings(
    library_dir: Path,
    slug: str,
    vectors: list[list[float]],
    descriptions: list[str] | None = None,
    tag_index: dict[str, list[int]] | None = None,
) -> None:
    """Create a film with a CLIP index + optional BM25 source files.

    ``descriptions`` (one per scene) populates ``scene_descriptions.json``,
    feeding the BM25 corpus. ``tag_index`` populates ``scene_tags.json``
    (used both by the per-film tag filter and as a BM25 token source).
    """
    md = library_dir / slug / "metadata"
    md.mkdir(parents=True)
    emb_dir = library_dir / slug / "embeddings"
    emb_dir.mkdir(parents=True)
    (library_dir / slug / "frames" / "keyframes").mkdir(parents=True)
    (library_dir / slug / "raw").mkdir()
    (library_dir / slug / "raw" / f"{slug}.mp4").write_bytes(b"")

    arr = np.array(vectors, dtype=np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)
    np.save(emb_dir / "keyframe_embeddings.npy", arr)

    kf_paths = [f"data/library/{slug}/frames/keyframes/{i}.jpg" for i in range(len(vectors))]
    mapping = {
        "total_vectors": len(vectors),
        "keyframe_paths": kf_paths,
        "scene_ids": list(range(len(vectors))),
        "keyframe_ids": list(range(len(vectors))),
    }
    (emb_dir / "index_mapping.json").write_text(json.dumps(mapping))
    (md / "keyframes_metadata.json").write_text(
        json.dumps(
            [
                {
                    "scene_id": i,
                    "filepath": kf_paths[i],
                    "start_time_s": float(i),
                }
                for i in range(len(vectors))
            ]
        )
    )
    if descriptions is not None:
        (md / "scene_descriptions.json").write_text(
            json.dumps(
                [{"scene_id": i, "description": descriptions[i]} for i in range(len(descriptions))]
            )
        )
    if tag_index is not None:
        (md / "scene_tags.json").write_text(json.dumps(tag_index))


def _cfg(library_dir: Path) -> object:
    return SimpleNamespace(paths=SimpleNamespace(library_dir=str(library_dir)))


@pytest.fixture()
def two_film_library_cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Build a 2-film library with both CLIP + BM25 source files.

    Film A (3 scenes) — distinct descriptions; "menina" only in scene 0.
    Film B (2 scenes) — "menina" appears in scene 1.

    Both films share the same 2D CLIP vector geometry so the stub
    embedder's [1, 0] query has the same score across them — keeping
    the test focused on RETRIEVER behaviour, not score arithmetic.

    The CLIP text-encoder is stubbed at module level so we don't pay
    the ~4 s real model load. ``_get_bm25_index_for_ctx`` is left
    untouched: it reads ``api.deps.get_config()`` for the BM25 sub-block,
    which the shared ``tmp_config`` fixture would patch — but this
    test deliberately exercises ``aggregate_search`` outside of any
    request context, so the loader's ``get_config`` call falls back to
    the real ``config/default.yaml`` (whose ``search.bm25`` block is
    populated, so the loader doesn't crash). The descriptions / tags
    on disk are the only thing that changes between tests.
    """
    library_dir = tmp_path / "library"
    library_dir.mkdir()
    register_film(library_dir, slug="a", title="A", year=2000, raw_filename="a.mp4")
    register_film(library_dir, slug="b", title="B", year=2001, raw_filename="b.mp4")
    # CLIP geometry: query [1, 0] vs each scene vector.
    # Film A: scene 0 = [0, 1] (cosine 0), scene 1 = [1, 0] (cosine 1),
    #         scene 2 = [0.5, 0.5] (cosine ~0.707).
    # Film B: scene 0 = [1, 0] (cosine 1), scene 1 = [0, 1] (cosine 0).
    _make_film_with_embeddings(
        library_dir,
        "a",
        [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]],
        descriptions=[
            "menina chorando na chuva",
            "homem caminhando na rua",
            "carro vermelho na estrada",
        ],
        tag_index={"outdoor": [0, 1, 2]},
    )
    _make_film_with_embeddings(
        library_dir,
        "b",
        [[1.0, 0.0], [0.0, 1.0]],
        descriptions=[
            "homem velho sentado no banco",
            "menina sorrindo no jardim",
        ],
        tag_index={"outdoor": [0, 1]},
    )

    class StubEmbedder:
        def encode_text(self, q: str) -> np.ndarray:
            return np.array([1.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(_AGGREGATE_MODULE, "_get_embedder", lambda cfg: StubEmbedder())
    return _cfg(library_dir)


# ── Tests ────────────────────────────────────────────────────────────────────


def test_aggregate_clip_mode_regression(two_film_library_cfg: object) -> None:
    """``retriever_mode="clip"`` reproduces the pre-M2 ordering exactly.

    Pin: calling ``aggregate_search`` with explicit ``retriever_mode="clip"``
    must produce byte-identical hits to the legacy default-kwarg path.
    Guards against D2 accidentally branching out of the legacy code path
    even when "clip" is requested.
    """
    clip_hits = aggregate_search(
        two_film_library_cfg,
        query="menina",
        modality="text",
        top_k=5,
        tags=[],
        min_similarity=0.0,
        retriever_mode="clip",
        sem_w=1.0,
        bm25_w=0.0,
    )
    legacy_hits = aggregate_search(
        two_film_library_cfg,
        query="menina",
        modality="text",
        top_k=5,
        tags=[],
        min_similarity=0.0,
    )
    assert [(h["film_slug"], h["scene_id"]) for h in clip_hits] == [
        (h["film_slug"], h["scene_id"]) for h in legacy_hits
    ]
    assert [h["score"] for h in clip_hits] == [h["score"] for h in legacy_hits]


def test_aggregate_hybrid_returns_results_from_multiple_films(
    two_film_library_cfg: object,
) -> None:
    """``retriever_mode="hybrid"`` returns hits from both films.

    The BM25 side surfaces film B scene 1 ("menina sorrindo no jardim")
    and film A scene 0 ("menina chorando na chuva"). The CLIP side
    (query [1, 0]) surfaces film A scene 1 and film B scene 0. After
    RRF fusion the union covers both films.
    """
    hits = aggregate_search(
        two_film_library_cfg,
        query="menina",
        modality="text",
        top_k=10,
        tags=[],
        min_similarity=0.0,
        retriever_mode="hybrid",
        sem_w=0.7,
        bm25_w=0.3,
    )
    assert isinstance(hits, list)
    assert len(hits) > 0
    slugs = {h["film_slug"] for h in hits}
    assert slugs == {"a", "b"}, f"hybrid hits must cover both films, got {slugs}"
    # Every hit carries the expected card shape.
    for h in hits:
        assert {"film_slug", "film_title", "scene_id", "score", "keyframe_path"} <= h.keys()


def test_aggregate_bm25_mode_returns_bm25_hits(two_film_library_cfg: object) -> None:
    """``retriever_mode="bm25"`` surfaces scenes by description-token match.

    The query "menina" appears only in scenes A:0 and B:1; neither has
    the highest CLIP score (which would be A:1 / B:0). A pure-BM25
    aggregate must return ONLY the "menina" scenes — proving the
    dispatcher is using BM25, not the CLIP cosine.
    """
    hits = aggregate_search(
        two_film_library_cfg,
        query="menina",
        modality="text",
        top_k=10,
        tags=[],
        min_similarity=0.0,
        retriever_mode="bm25",
        sem_w=0.0,
        bm25_w=1.0,
    )
    assert len(hits) >= 1, "BM25 must surface the 'menina' scenes"
    bm25_pairs = {(h["film_slug"], h["scene_id"]) for h in hits}
    # Every BM25 hit must be from a scene whose description contains
    # "menina" — never the high-CLIP-score scenes (A:1, B:0).
    expected_pairs = {("a", 0), ("b", 1)}
    assert (
        bm25_pairs <= expected_pairs
    ), f"BM25-mode hits leaked non-'menina' scenes: {bm25_pairs - expected_pairs}"


def test_aggregate_bm25_mode_respects_tag_filter(
    two_film_library_cfg: object,
) -> None:
    """A ``tags=[...]`` filter must apply to BM25 hits too.

    Without this, ``?retriever=bm25&tags=outdoor`` would silently
    ignore the tag selection (regression risk: CLIP path was the only
    one originally tag-filtering, easy to forget for the new branches).

    We pin film B's ``outdoor`` tag to only scene 0; the only "menina"
    scene in B (scene 1) is no longer outdoor. A BM25-mode search for
    "menina" with ``tags=["outdoor"]`` must drop B:1 and keep only A:0
    (which is still outdoor).
    """
    # Re-write film B's tag index to a narrower set.
    library_dir = Path(two_film_library_cfg.paths.library_dir)  # type: ignore[attr-defined]
    (library_dir / "b" / "metadata" / "scene_tags.json").write_text(
        json.dumps({"outdoor": [0]})  # was [0, 1] — B:1 no longer outdoor
    )

    hits = aggregate_search(
        two_film_library_cfg,
        query="menina",
        modality="text",
        top_k=10,
        tags=["outdoor"],
        min_similarity=0.0,
        retriever_mode="bm25",
        sem_w=0.0,
        bm25_w=1.0,
    )
    pairs = {(h["film_slug"], h["scene_id"]) for h in hits}
    # A:0 is "menina" + "outdoor" → kept. B:1 is "menina" but NOT "outdoor"
    # under the rewritten tag index → must be dropped.
    assert ("a", 0) in pairs, "A:0 (menina + outdoor) must survive the tag filter"
    assert (
        "b",
        1,
    ) not in pairs, "B:1 (menina, NOT outdoor) leaked through — tag filter not applied to BM25 hits"


@pytest.fixture()
def degeneracy_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Two 3-doc films, each contributing a rank-1 CLIP + rank-1 BM25 scene.

    With ≥3 docs per film, BM25 IDF on a one-doc match is strictly
    positive — so the per-film top-1 in each film has score
    ``sem_w/(rrf_k+1) + bm25_w/(rrf_k+1) = 1/(rrf_k+1)``. Cross-film
    these two scenes would tie under per-film-RRF + raw-score sort
    (the bug). Global RRF must break the tie by global-rank.
    """
    library_dir = tmp_path / "library"
    library_dir.mkdir()
    register_film(library_dir, slug="a", title="A", year=2000, raw_filename="a.mp4")
    register_film(library_dir, slug="b", title="B", year=2001, raw_filename="b.mp4")
    _make_film_with_embeddings(
        library_dir,
        "a",
        [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]],
        descriptions=["apple", "banana", "cherry"],
    )
    _make_film_with_embeddings(
        library_dir,
        "b",
        [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
        descriptions=["banana", "durian", "elderberry"],
    )

    class StubEmbedder:
        def encode_text(self, q: str) -> np.ndarray:
            return np.array([1.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(_AGGREGATE_MODULE, "_get_embedder", lambda cfg: StubEmbedder())
    return _cfg(library_dir)


def test_aggregate_hybrid_no_cross_film_top_score_ties(degeneracy_fixture: object) -> None:
    """Cross-film hybrid ordering must use GLOBAL RRF, not per-film RRF + sort.

    With per-film RRF, every film's per-film rank-1 scene tops out at
    ``sem_w/(rrf_k+1) + bm25_w/(rrf_k+1) = 1/(rrf_k+1)``. With two
    films both contributing a rank-1 scene from BOTH the CLIP side
    (A:1, B:0 each cosine 1) AND the BM25 side (banana hits in
    each), the cross-film naive sort sees two identical-score
    scenes at the top — they TIE, and which one displays first is
    determined by film-iteration order, not signal strength.

    Global RRF assigns ranks across the union of per-film hits;
    only ONE scene can be globally rank 1 per side, so the top
    results have distinct scores reflecting their global standing.
    Failure mode under per-film RRF: ``hits[0]['score'] == hits[1]['score']``.
    """
    hits = aggregate_search(
        degeneracy_fixture,
        query="banana",
        modality="text",
        top_k=6,
        tags=[],
        min_similarity=0.0,
        retriever_mode="hybrid",
        sem_w=0.7,
        bm25_w=0.3,
    )
    assert len(hits) >= 2, "fixture must produce at least 2 hybrid hits"
    top_keys = {(h["film_slug"], h["scene_id"]) for h in hits[:2]}
    assert top_keys == {("a", 1), ("b", 0)}, (
        f"top-2 must still be the two combined-signal scenes, got {top_keys}"
    )
    top, second = hits[0]["score"], hits[1]["score"]
    assert top > second, (
        f"hybrid cross-film ordering must be strict: top={top} second={second} — "
        f"per-film RRF is degenerate (each per-film rank-1 yields the same score), "
        f"global RRF is required to disambiguate"
    )


def test_aggregate_search_honours_rrf_k(two_film_library_cfg: object) -> None:
    """``aggregate_search`` accepts an ``rrf_k`` kwarg that reaches ``fuse_rrf``.

    Pre-fix: ``aggregate_search`` hard-coded ``DEFAULT_RRF_K`` in its
    fuse_rrf call, so ``config/default.yaml`` → ``search.bm25.rrf_k``
    was a dead knob. We verify here that running the same query with
    rrf_k=1 yields a top-1 fused score >5× larger than with rrf_k=60 —
    same ordering, sharper magnitude. Once the route plumbs the config
    value through, an operator changing rrf_k in YAML actually changes
    fusion behaviour.
    """
    common = dict(
        query="menina",
        modality="text",
        top_k=5,
        tags=[],
        min_similarity=0.0,
        retriever_mode="hybrid",
        sem_w=0.7,
        bm25_w=0.3,
    )
    hits_default = aggregate_search(two_film_library_cfg, **common, rrf_k=60)
    hits_small = aggregate_search(two_film_library_cfg, **common, rrf_k=1)
    assert hits_default and hits_small, "fixture must produce hits in both runs"
    top_default = hits_default[0]["score"]
    top_small = hits_small[0]["score"]
    assert top_small > top_default * 5, (
        f"rrf_k=1 must yield much larger fused score than rrf_k=60: "
        f"small={top_small} default={top_default}"
    )


def test_aggregate_hybrid_falls_back_to_clip_when_bm25_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A film with no BM25 source files must silently degrade to CLIP.

    ``BM25Index.build`` returns ``BM25Index(model=None, ...)`` when the
    corpus is empty. The hybrid path treats this as "clip-only for this
    film" rather than raising, so the cross-film merge still produces
    hits from films that DO have descriptions, plus clip-only hits
    from those that don't.
    """
    library_dir = tmp_path / "library"
    library_dir.mkdir()
    register_film(library_dir, slug="a", title="A", year=2000, raw_filename="a.mp4")
    # No descriptions / no tags → BM25 corpus is empty.
    _make_film_with_embeddings(library_dir, "a", [[1.0, 0.0], [0.0, 1.0]])

    class StubEmbedder:
        def encode_text(self, q: str) -> np.ndarray:
            return np.array([1.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(_AGGREGATE_MODULE, "_get_embedder", lambda cfg: StubEmbedder())

    hits = aggregate_search(
        _cfg(library_dir),
        query="menina",
        modality="text",
        top_k=5,
        tags=[],
        min_similarity=0.0,
        retriever_mode="hybrid",
        sem_w=0.7,
        bm25_w=0.3,
    )
    # Empty BM25 corpus → film degrades to clip-only; the cosine top
    # hit (scene 0, score 1.0) still surfaces.
    assert len(hits) >= 1
    assert hits[0]["scene_id"] == 0
    assert hits[0]["film_slug"] == "a"
