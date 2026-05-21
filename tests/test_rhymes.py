"""Tests for cinemateca.rhymes — cross-film cosine kNN over CLIP keyframes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cinemateca.rhymes import Rhyme, find_rhymes


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
