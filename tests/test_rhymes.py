"""Tests for cinemateca.rhymes — cross-film cosine kNN over CLIP keyframes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cinemateca.rhymes import Rhyme, find_rhymes
from cinemateca.rhymes.algorithm import _extract_scene_ids


def _write_film(
    library_dir: Path,
    slug: str,
    vectors: np.ndarray,
    scene_ids: list[int],
) -> None:
    """Lay out a minimal per-film embeddings directory the way Task 20 expects."""
    d = library_dir / slug / "embeddings"
    d.mkdir(parents=True)
    np.save(d / "keyframe_embeddings.npy", vectors)
    (d / "index_mapping.json").write_text(
        json.dumps({"scene_ids": scene_ids, "dimension": int(vectors.shape[1])})
    )


def test_find_rhymes_returns_topn_cross_film(tmp_path: Path) -> None:
    """Anchor in film_a should pull top-N matches strictly from film_b."""
    library_dir = tmp_path / "library"

    a_vecs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    b_vecs = np.array([[1, 0.05, 0], [0.2, 0.9, 0], [0.0, 0.1, 0.99]], dtype=np.float32)
    a_norm = a_vecs / np.linalg.norm(a_vecs, axis=1, keepdims=True)
    b_norm = b_vecs / np.linalg.norm(b_vecs, axis=1, keepdims=True)
    _write_film(library_dir, "film_a", a_norm, [1, 2, 3])
    _write_film(library_dir, "film_b", b_norm, [1, 2, 3])

    results = find_rhymes(
        library_dir=library_dir,
        anchor_slug="film_a",
        anchor_scene_id=1,
        top_n=3,
    )

    assert all(isinstance(r, Rhyme) for r in results)
    # Cross-film constraint: anchor's own film must not appear.
    assert all(r.film_slug == "film_b" for r in results)
    # film_b row 0 ([1, 0.05, 0]) aligns with anchor [1, 0, 0]; should top.
    assert results[0].scene_id == 1
    assert results[0].score > 0.99


def test_find_rhymes_returns_empty_when_no_index(tmp_path: Path) -> None:
    """Missing anchor embeddings index degrades gracefully to []."""
    results = find_rhymes(
        library_dir=tmp_path / "library",
        anchor_slug="missing",
        anchor_scene_id=1,
        top_n=3,
    )
    assert results == []


def test_extract_scene_ids_synthetic_shape() -> None:
    """The ``scene_ids`` shape (test fixtures) is returned verbatim as ints."""
    assert _extract_scene_ids({"scene_ids": [1, 2, 3]}) == [1, 2, 3]
    # Strings coerce to ints (JSON round-trips can change types).
    assert _extract_scene_ids({"scene_ids": ["4", "5"]}) == [4, 5]


def test_extract_scene_ids_production_shape() -> None:
    """The ``keyframe_paths`` shape (PySceneDetect) parses ``Scene-NNN`` from filenames.

    Multiple keyframes per scene collapse to repeated scene ids in the
    list — row-count alignment with the embeddings matrix is preserved
    by design.
    """
    mapping = {
        "keyframe_paths": [
            "data/library/jeca_tatu/frames/scenes/keyframes_content/Mazzaropi-Jeca_Tatu-Scene-001-01.jpg",
            "data/library/jeca_tatu/frames/scenes/keyframes_content/Mazzaropi-Jeca_Tatu-Scene-001-02.jpg",
            "data/library/jeca_tatu/frames/scenes/keyframes_content/Mazzaropi-Jeca_Tatu-Scene-002-01.jpg",
            "data/library/jeca_tatu/frames/scenes/keyframes_content/Mazzaropi-Jeca_Tatu-Scene-007-03.jpg",
        ]
    }
    assert _extract_scene_ids(mapping) == [1, 1, 2, 7]


def test_extract_scene_ids_unparseable_filenames_get_minus_one() -> None:
    """Filenames that do not match ``Scene-NNN`` emit ``-1`` to keep row alignment."""
    mapping = {"keyframe_paths": ["random_name.jpg", "another-Scene-005-01.jpg"]}
    assert _extract_scene_ids(mapping) == [-1, 5]


def test_extract_scene_ids_unknown_shape_returns_empty() -> None:
    """A mapping with neither known key returns ``[]`` — callers treat as corrupt."""
    assert _extract_scene_ids({}) == []
    assert _extract_scene_ids({"total_vectors": 42, "model": "CLIP"}) == []


def test_find_rhymes_with_keyframe_paths_mapping(tmp_path: Path) -> None:
    """The production ``keyframe_paths`` shape works end-to-end through ``find_rhymes``.

    Each film carries 3 keyframes (one scene with 3 keyframes for the
    anchor, two scenes with 1 + 2 keyframes for the candidate). Scene
    ids derive from the ``Scene-NNN`` portion of the filename.
    """
    library_dir = tmp_path / "library"

    # film_a: 3 keyframes, all scene 1 (anchor's scene).
    a_vecs = np.array([[1, 0, 0], [0.9, 0.1, 0], [0.95, 0, 0.05]], dtype=np.float32)
    a_norm = a_vecs / np.linalg.norm(a_vecs, axis=1, keepdims=True)
    _write_film_keyframe_paths(
        library_dir,
        "film_a",
        a_norm,
        [
            "Anchor-Film-Scene-001-01.jpg",
            "Anchor-Film-Scene-001-02.jpg",
            "Anchor-Film-Scene-001-03.jpg",
        ],
    )
    # film_b: 3 keyframes — scene 5 (aligned), scene 6, scene 7 (off).
    b_vecs = np.array([[1, 0.05, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    b_norm = b_vecs / np.linalg.norm(b_vecs, axis=1, keepdims=True)
    _write_film_keyframe_paths(
        library_dir,
        "film_b",
        b_norm,
        [
            "Other-Film-Scene-005-01.jpg",
            "Other-Film-Scene-006-01.jpg",
            "Other-Film-Scene-007-01.jpg",
        ],
    )

    results = find_rhymes(
        library_dir=library_dir,
        anchor_slug="film_a",
        anchor_scene_id=1,
        top_n=3,
    )

    assert len(results) == 3
    assert all(r.film_slug == "film_b" for r in results)
    # film_b row 0 (scene 5, [1, 0.05, 0]) aligns with anchor [1, 0, 0].
    assert results[0].scene_id == 5
    assert results[0].score > 0.99


def _write_film_keyframe_paths(
    library_dir: Path,
    slug: str,
    vectors: np.ndarray,
    keyframe_paths: list[str],
) -> None:
    """Lay out a per-film embeddings dir with the production ``keyframe_paths`` shape."""
    d = library_dir / slug / "embeddings"
    d.mkdir(parents=True)
    np.save(d / "keyframe_embeddings.npy", vectors)
    (d / "index_mapping.json").write_text(
        json.dumps(
            {
                "model": "CLIP ViT-B-32 (openai)",
                "dimension": int(vectors.shape[1]),
                "total_vectors": int(vectors.shape[0]),
                "normalized": True,
                "keyframe_paths": keyframe_paths,
            }
        )
    )
