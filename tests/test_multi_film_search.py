"""Multi-film aggregate context / search tests."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from api.services.catalog import build_scenes_context_aggregate
from api.services.search import aggregate_search
from cinemateca.library import register_film


def _make_film(library_dir: Path, slug: str, scene_ids: list[int]) -> None:
    """Create a minimal per-film layout with N scenes + a tag index entry."""
    md = library_dir / slug / "metadata"
    md.mkdir(parents=True)
    (library_dir / slug / "frames" / "keyframes").mkdir(parents=True)
    (library_dir / slug / "raw").mkdir()
    (library_dir / slug / "raw" / f"{slug}.mp4").write_bytes(b"")
    kf_meta = [
        {
            "scene_id": sid,
            "filepath": f"data/library/{slug}/frames/keyframes/{sid}.jpg",
            "start_time_s": float(sid),
        }
        for sid in scene_ids
    ]
    (md / "keyframes_metadata.json").write_text(json.dumps(kf_meta))
    (md / "scene_tags.json").write_text(
        json.dumps({"outdoor": scene_ids})
    )


def _cfg(library_dir: Path) -> object:
    return SimpleNamespace(paths=SimpleNamespace(library_dir=str(library_dir)))


def test_aggregate_scenes_context_merges_films(tmp_path: Path) -> None:
    library_dir = tmp_path / "library"
    library_dir.mkdir()
    register_film(library_dir, slug="a", title="A", year=2000, raw_filename="a.mp4")
    register_film(library_dir, slug="b", title="B", year=2001, raw_filename="b.mp4")
    _make_film(library_dir, "a", [0, 1, 2])
    _make_film(library_dir, "b", [0, 1])

    ctx = build_scenes_context_aggregate(_cfg(library_dir))

    assert len(ctx["cards"]) == 5
    assert {c["film_slug"] for c in ctx["cards"]} == {"a", "b"}
    assert "outdoor" in ctx["available_tags"]
    assert ctx["no_data"] is False


def test_aggregate_scenes_context_empty_when_no_films(tmp_path: Path) -> None:
    library_dir = tmp_path / "library"
    library_dir.mkdir()
    ctx = build_scenes_context_aggregate(_cfg(library_dir))
    assert ctx["cards"] == []
    assert ctx["no_data"] is True


def test_aggregate_tolerates_unprocessed_film(tmp_path: Path) -> None:
    """A registered film with no metadata/ dir contributes 0 cards (no crash)."""
    library_dir = tmp_path / "library"
    library_dir.mkdir()
    register_film(library_dir, slug="a", title="A", year=2000, raw_filename="a.mp4")
    register_film(library_dir, slug="unproc", title="Unprocessed", year=2001, raw_filename="u.mp4")
    _make_film(library_dir, "a", [0, 1, 2])
    # Create just the film_dir for unproc (so FilmContext.for_film doesn't raise),
    # but skip metadata/ — emulates a freshly-registered, never-processed film.
    (library_dir / "unproc" / "raw").mkdir(parents=True)
    (library_dir / "unproc" / "raw" / "u.mp4").write_bytes(b"")

    ctx = build_scenes_context_aggregate(_cfg(library_dir))

    assert len(ctx["cards"]) == 3  # only film "a"
    assert {c["film_slug"] for c in ctx["cards"]} == {"a"}
    assert ctx["no_data"] is False


# ── aggregate_search tests ────────────────────────────────────────────────────

def _make_film_with_embeddings(
    library_dir: Path, slug: str, vectors: list[list[float]]
) -> None:
    """Create a film with a tiny CLIP index of ``len(vectors)`` scenes."""
    md = library_dir / slug / "metadata"
    md.mkdir(parents=True)
    emb_dir = library_dir / slug / "embeddings"
    emb_dir.mkdir(parents=True)
    (library_dir / slug / "frames" / "keyframes").mkdir(parents=True)
    (library_dir / slug / "raw").mkdir()
    (library_dir / slug / "raw" / f"{slug}.mp4").write_bytes(b"")

    arr = np.array(vectors, dtype=np.float32)
    # L2-normalise so cosine == dot product
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)
    np.save(emb_dir / "keyframe_embeddings.npy", arr)

    # Dict format expected by OpenClipEmbedder.load
    kf_paths = [
        f"data/library/{slug}/frames/keyframes/{i}.jpg"
        for i in range(len(vectors))
    ]
    mapping = {
        "total_vectors": len(vectors),
        "keyframe_paths": kf_paths,
        "scene_ids": list(range(len(vectors))),
        "keyframe_ids": list(range(len(vectors))),
    }
    (emb_dir / "index_mapping.json").write_text(json.dumps(mapping))
    # Stub keyframes_metadata.json so build_cards in the result-merger works:
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


def test_aggregate_search_empty_library_does_not_load_embedder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """aggregate_search returns [] immediately when the library has no films.

    _get_embedder must NOT be called — loading the CLIP model (~4 s) is
    expensive and pointless when there are no indexed films to search.
    """
    library_dir = tmp_path / "library"
    library_dir.mkdir()
    # No register_film call → scan_library returns [].

    def _should_not_load(cfg: object) -> object:
        raise AssertionError(
            "_get_embedder was called on an empty library — CLIP eager-load bug"
        )

    monkeypatch.setattr("api.services.search._get_embedder", _should_not_load)

    result = aggregate_search(
        _cfg(library_dir), query="anything", modality="text", top_k=10
    )
    assert result == []


def test_aggregate_text_search_returns_results_from_both_films(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """aggregate_search merges per-film results by score, top-K overall."""
    library_dir = tmp_path / "library"
    library_dir.mkdir()
    register_film(library_dir, slug="a", title="A", year=2000, raw_filename="a.mp4")
    register_film(library_dir, slug="b", title="B", year=2001, raw_filename="b.mp4")
    # Film A: best match at index 1 (score 1.0); film B: best match at index 0.
    # Using [0.0, 1.0] as the low-scoring A scene (score 0.0 vs query [1,0])
    # so the top-2 overall are A[1,0] and B[1,0], one from each film.
    _make_film_with_embeddings(library_dir, "a", [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
    _make_film_with_embeddings(library_dir, "b", [[1.0, 0.0], [0.0, 1.0]])

    # Stub the CLIP text-encoder so we don't need a real model in tests.
    class StubEmbedder:
        def encode_text(self, q: str) -> np.ndarray:
            return np.array([1.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(
        "api.services.search._get_embedder", lambda cfg: StubEmbedder()
    )

    results = aggregate_search(
        _cfg(library_dir), query="anything", modality="text", top_k=2
    )

    assert len(results) == 2
    # Both top-2 results are the "[1.0, 0.0]" matches (score 1.0): one from
    # film A (index 1) and one from film B (index 0). A's [0.5, 0.5] at
    # score ~0.707 is rank-3 and is excluded by top_k=2.
    slugs_in_top2 = {r["film_slug"] for r in results}
    assert slugs_in_top2 == {"a", "b"}
    assert all(abs(r["score"] - 1.0) < 1e-6 for r in results)


def _write_tag_index(library_dir: Path, slug: str, tag_to_scene_ids: dict[str, list[int]]) -> None:
    """Overwrite a film's ``scene_tags.json`` with the supplied tag index."""
    (library_dir / slug / "metadata" / "scene_tags.json").write_text(
        json.dumps(tag_to_scene_ids)
    )


def test_aggregate_search_filters_by_tags_per_film(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``aggregate_search(..., tags=[t])`` restricts hits to scenes whose
    ``scene_id`` is in EVERY selected tag's list, mirroring
    ``SemanticSearch.combined``'s per-film semantics.

    Setup: film A has tag ``floor`` covering only scene_id=0 (the LOW-score
    scene against the query); without the tag filter, scene_id=1 (score 1.0)
    would win. With ``tags=["floor"]``, scene_id=1 is excluded — the only
    hit is scene_id=0.
    """
    library_dir = tmp_path / "library"
    library_dir.mkdir()
    register_film(library_dir, slug="a", title="A", year=2000, raw_filename="a.mp4")
    _make_film_with_embeddings(library_dir, "a", [[0.0, 1.0], [1.0, 0.0]])
    _write_tag_index(library_dir, "a", {"floor": [0]})

    class StubEmbedder:
        def encode_text(self, q: str) -> np.ndarray:
            return np.array([1.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(
        "api.services.search._get_embedder", lambda cfg: StubEmbedder()
    )

    results = aggregate_search(
        _cfg(library_dir), query="anything", modality="text", top_k=8,
        tags=["floor"],
    )

    assert len(results) == 1
    assert results[0]["scene_id"] == 0
    assert results[0]["film_slug"] == "a"


def test_aggregate_search_skips_films_missing_selected_tag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A film that lacks ANY selected tag contributes zero hits, leaving
    the merged result list scoped to films that have all the requested tags."""
    library_dir = tmp_path / "library"
    library_dir.mkdir()
    register_film(library_dir, slug="a", title="A", year=2000, raw_filename="a.mp4")
    register_film(library_dir, slug="b", title="B", year=2001, raw_filename="b.mp4")
    _make_film_with_embeddings(library_dir, "a", [[1.0, 0.0]])
    _make_film_with_embeddings(library_dir, "b", [[1.0, 0.0]])
    # Only film A carries the ``floor`` tag.
    _write_tag_index(library_dir, "a", {"floor": [0]})
    _write_tag_index(library_dir, "b", {"sky": [0]})

    class StubEmbedder:
        def encode_text(self, q: str) -> np.ndarray:
            return np.array([1.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(
        "api.services.search._get_embedder", lambda cfg: StubEmbedder()
    )

    results = aggregate_search(
        _cfg(library_dir), query="anything", modality="text", top_k=8,
        tags=["floor"],
    )

    assert {r["film_slug"] for r in results} == {"a"}


def test_aggregate_search_no_tags_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Passing ``tags=None`` / ``tags=[]`` is a no-op — the loader does
    NOT touch each film's tag_index, preserving the prior fast path."""
    library_dir = tmp_path / "library"
    library_dir.mkdir()
    register_film(library_dir, slug="a", title="A", year=2000, raw_filename="a.mp4")
    _make_film_with_embeddings(library_dir, "a", [[1.0, 0.0]])

    class StubEmbedder:
        def encode_text(self, q: str) -> np.ndarray:
            return np.array([1.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(
        "api.services.search._get_embedder", lambda cfg: StubEmbedder()
    )

    results_default = aggregate_search(
        _cfg(library_dir), query="anything", modality="text", top_k=8
    )
    results_empty = aggregate_search(
        _cfg(library_dir), query="anything", modality="text", top_k=8, tags=[],
    )
    assert len(results_default) == 1
    assert len(results_empty) == 1
    assert results_default[0]["scene_id"] == results_empty[0]["scene_id"] == 0


def test_build_search_context_aggregate_unions_and_filters(
    tmp_path: Path,
) -> None:
    """The aggregate context unions tag-index keys across films AND drops
    degenerate-looking strings (sentence fragments, repeated tokens, etc.).
    """
    from api.services.search import build_search_context_aggregate

    library_dir = tmp_path / "library"
    library_dir.mkdir()
    register_film(library_dir, slug="a", title="A", year=2000, raw_filename="a.mp4")
    register_film(library_dir, slug="b", title="B", year=2001, raw_filename="b.mp4")
    (library_dir / "a" / "metadata").mkdir(parents=True)
    (library_dir / "b" / "metadata").mkdir(parents=True)
    (library_dir / "a" / "raw").mkdir()
    (library_dir / "b" / "raw").mkdir()
    _write_tag_index(library_dir, "a", {
        "dia": [0, 1],
        "exterior": [0],
        "fence-gate-gate-gate-gate-gate-gate": [1],  # garbage → drop
    })
    _write_tag_index(library_dir, "b", {
        "noite": [0],
        "exterior": [0],  # also in A → unioned, not duplicated
        "a-rural-field-with-a-fence.": [0],  # full-caption garbage → drop
    })

    out = build_search_context_aggregate(_cfg(library_dir))

    assert out["available_tags"] == sorted(["dia", "exterior", "noite"])


def test_aggregate_search_includes_timecode_per_hit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each hit carries a SMPTE ``timecode`` computed from its film's
    ``keyframes_metadata.json``. The scene at ``start_time_s=0`` gets ""
    (template hides the span); a positive ``start_time_s`` produces a
    non-empty SMPTE string.
    """
    library_dir = tmp_path / "library"
    library_dir.mkdir()
    register_film(library_dir, slug="a", title="A", year=2000, raw_filename="a.mp4")
    # _make_film_with_embeddings writes scene_id i → start_time_s=i; scene 0
    # has start_time_s=0 (timecode ""), scene 1 has start_time_s=1.0
    # (timecode non-empty). The CLIP stub picks vector [1,0] (scene 1) as
    # the top match.
    _make_film_with_embeddings(library_dir, "a", [[0.0, 1.0], [1.0, 0.0]])

    class StubEmbedder:
        def encode_text(self, q: str) -> np.ndarray:
            return np.array([1.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(
        "api.services.search._get_embedder", lambda cfg: StubEmbedder()
    )

    results = aggregate_search(
        _cfg(library_dir), query="anything", modality="text", top_k=2
    )

    by_scene = {r["scene_id"]: r for r in results}
    # Scene 0 (start_time_s=0) → empty timecode by the "> 0" guard the
    # per-film path also uses (matches results_to_dicts behaviour).
    assert by_scene[0]["timecode"] == ""
    # Scene 1 (start_time_s=1.0) → non-empty SMPTE. Exact value depends on
    # derive_fps's fallback (24.0); just assert it's a populated string.
    assert by_scene[1]["timecode"] != ""
    assert ":" in by_scene[1]["timecode"]


def test_aggregate_search_drops_below_min_similarity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``min_similarity`` filters per-hit scores BEFORE the top-K merge.

    Setup: vector [0.5, 0.5] (normalised) yields cosine 0.707 against
    query [1, 0]; vector [1.0, 0.0] yields 1.0. With
    ``min_similarity=0.8`` only the perfect-match hit survives.
    """
    library_dir = tmp_path / "library"
    library_dir.mkdir()
    register_film(library_dir, slug="a", title="A", year=2000, raw_filename="a.mp4")
    _make_film_with_embeddings(library_dir, "a", [[1.0, 0.0], [0.5, 0.5]])

    class StubEmbedder:
        def encode_text(self, q: str) -> np.ndarray:
            return np.array([1.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(
        "api.services.search._get_embedder", lambda cfg: StubEmbedder()
    )

    # No threshold → both hits returned.
    no_floor = aggregate_search(
        _cfg(library_dir), query="x", modality="text", top_k=8,
    )
    assert len(no_floor) == 2

    # 0.8 floor → only the score=1.0 hit survives.
    above = aggregate_search(
        _cfg(library_dir), query="x", modality="text", top_k=8,
        min_similarity=0.8,
    )
    assert len(above) == 1
    assert above[0]["scene_id"] == 0
    assert above[0]["score"] >= 0.8

    # Floor above every score → empty.
    none = aggregate_search(
        _cfg(library_dir), query="x", modality="text", top_k=8,
        min_similarity=1.5,
    )
    assert none == []
