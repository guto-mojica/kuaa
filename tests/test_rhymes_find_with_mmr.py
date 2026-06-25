"""find_rhymes() honours lambda_diversity + k_candidates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kuaa.rhymes import find_rhymes


def _make_library(tmp_path: Path) -> Path:
    """Two films: 'anchor' (5 scenes) and 'other' (5 scenes).
    'other' scenes 1-3 are near-duplicates; scene 4 is diverse."""
    lib = tmp_path / "library"
    for slug, vectors in [
        ("anchor", np.eye(5, 4, dtype="float32")),  # 5 scenes in 4-D
        (
            "other",
            np.vstack(
                [
                    [0.9, 0.1, 0.0, 0.0],  # near-dup #1
                    [0.91, 0.09, 0.0, 0.0],  # near-dup #2
                    [0.92, 0.08, 0.0, 0.0],  # near-dup #3
                    [0.0, 0.0, 1.0, 0.0],  # diverse
                    [0.0, 0.5, 0.5, 0.0],  # somewhat diverse
                ]
            ).astype("float32"),
        ),
    ]:
        film = lib / slug
        (film / "embeddings").mkdir(parents=True)
        # L2-normalise each row.
        v = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        np.save(film / "embeddings" / "keyframe_embeddings.npy", v)
        (film / "embeddings" / "index_mapping.json").write_text(
            json.dumps({"scene_ids": list(range(1, len(v) + 1))})
        )
    return lib


def test_find_rhymes_lambda_one_returns_pure_knn_top_n(tmp_path: Path) -> None:
    lib = _make_library(tmp_path)
    out = find_rhymes(
        library_dir=lib,
        anchor_slug="anchor",
        anchor_scene_id=1,
        top_n=3,
        lambda_diversity=1.0,  # NEW
        k_candidates=5,  # NEW
    )
    # Anchor scene 1 is [1,0,0,0]. Pure kNN over 'other' ranks by cosine:
    # near-dups #1/#2/#3 score ~0.99 (after L2-normalisation, #3 > #2 > #1
    # because the [0.92, 0.08] row sits closer to the x-axis once
    # unit-normalised); diverse #4 = 0; somewhat #5 = 0.
    # Top-3 should be the 3 near-dups — order is the deterministic
    # post-normalisation descending-cosine ranking.
    assert [r.scene_id for r in out] == [3, 2, 1]


def test_find_rhymes_lambda_half_diversifies_top_n(tmp_path: Path) -> None:
    lib = _make_library(tmp_path)
    out = find_rhymes(
        library_dir=lib,
        anchor_slug="anchor",
        anchor_scene_id=1,
        top_n=3,
        lambda_diversity=0.5,
        k_candidates=5,
    )
    # MMR should knock out at least one near-dup in favour of #4 or #5.
    scene_ids = {r.scene_id for r in out}
    assert (4 in scene_ids) or (5 in scene_ids), (
        f"MMR failed to break near-dup cluster: {scene_ids}"
    )


def test_find_rhymes_backward_compat_default_args(tmp_path: Path) -> None:
    """Calling find_rhymes without the new kwargs reproduces the M1
    stub behaviour (pure kNN, top_n=8)."""
    lib = _make_library(tmp_path)
    out = find_rhymes(
        library_dir=lib,
        anchor_slug="anchor",
        anchor_scene_id=1,
        top_n=3,
    )
    # Default lambda_diversity=1.0 → pure relevance; same descending-cosine
    # order as the explicit-lambda=1.0 test above.
    assert [r.scene_id for r in out] == [3, 2, 1]
