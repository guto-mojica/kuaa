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
