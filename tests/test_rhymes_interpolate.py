"""Interpolation between two anchors — retrieval through the space between.

Synthetic embeddings throughout: the geometry is the thing under test, and
a fixture library keeps it verifiable from inputs alone.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from kuaa.rhymes.interpolate import (
    DEFAULT_STEPS,
    MAX_ANCHOR_COSINE,
    AnchorsTooSimilar,
    interpolate_path,
    novelty_rate,
    slerp,
)


def _write_film(library_dir, slug: str, vectors: list[list[float]]) -> None:
    emb_dir = library_dir / slug / "embeddings"
    emb_dir.mkdir(parents=True)
    arr = np.asarray(vectors, dtype="float32")
    arr = arr / np.linalg.norm(arr, axis=1, keepdims=True)
    np.save(emb_dir / "keyframe_embeddings.npy", arr)
    (emb_dir / "index_mapping.json").write_text(
        json.dumps({"scene_ids": list(range(1, len(vectors) + 1))})
    )


@pytest.fixture
def library(tmp_path):
    """Four scenes on a 2-D circle: A-ish, between, B-ish, and far away."""
    lib = tmp_path / "library"
    lib.mkdir()
    _write_film(
        lib,
        "alpha",
        [
            [1.0, 0.0],  # scene 1 — sits on anchor A
            [1.0, 1.0],  # scene 2 — the interpolant, between A and B
            [0.0, 1.0],  # scene 3 — sits on anchor B
            [-1.0, -0.2],  # scene 4 — opposite, should never surface
        ],
    )
    return lib


# ── slerp geometry ───────────────────────────────────────────────────────────


def test_slerp_endpoints_are_the_anchors() -> None:
    a = np.array([1.0, 0.0], dtype="float32")
    b = np.array([0.0, 1.0], dtype="float32")
    assert np.allclose(slerp(a, b, 0.0), a, atol=1e-6)
    assert np.allclose(slerp(a, b, 1.0), b, atol=1e-6)


def test_slerp_stays_on_the_unit_sphere() -> None:
    """Linear interpolation leaves the sphere; SigLIP vectors live on it."""
    a = np.array([1.0, 0.0], dtype="float32")
    b = np.array([-0.9, 0.4], dtype="float32")
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert np.isclose(np.linalg.norm(slerp(a, b, t)), 1.0, atol=1e-5)


def test_slerp_midpoint_bisects_the_angle() -> None:
    a = np.array([1.0, 0.0], dtype="float32")
    b = np.array([0.0, 1.0], dtype="float32")
    mid = slerp(a, b, 0.5)
    assert np.isclose(float(mid @ a), float(mid @ b), atol=1e-6)


def test_slerp_handles_parallel_anchors_without_blowing_up() -> None:
    """The slerp denominator vanishes when anchors are parallel."""
    a = np.array([1.0, 0.0], dtype="float32")
    out = slerp(a, a.copy(), 0.5)
    assert np.isclose(np.linalg.norm(out), 1.0, atol=1e-5)


# ── the distance gate ────────────────────────────────────────────────────────


def test_near_identical_anchors_are_refused(library) -> None:
    """Below the gate there is no space between the anchors to search.

    This is a failure mode, not a preference — hence a fixed gate rather
    than a slider.
    """
    a = np.array([1.0, 0.0], dtype="float32")
    b = np.array([1.0, 0.05], dtype="float32")
    with pytest.raises(AnchorsTooSimilar) as excinfo:
        interpolate_path(library, a, b)
    assert excinfo.value.cosine >= MAX_ANCHOR_COSINE


def test_distance_gate_can_be_disabled_for_experiments(library) -> None:
    a = np.array([1.0, 0.0], dtype="float32")
    b = np.array([1.0, 0.05], dtype="float32")
    assert interpolate_path(library, a, b, enforce_distance=False)


# ── retrieval through the gap ────────────────────────────────────────────────


def test_midpoint_surfaces_the_scene_neither_anchor_would(library) -> None:
    """The whole point: scene 2 lies between A and B and belongs to neither."""
    a = np.array([1.0, 0.0], dtype="float32")
    b = np.array([0.0, 1.0], dtype="float32")
    results = interpolate_path(library, a, b, steps=(0.5,), top_k=1)
    assert [(r.scene_id, r.novel) for r in results] == [(2, True)]


def test_endpoint_scenes_are_not_marked_novel(library) -> None:
    a = np.array([1.0, 0.0], dtype="float32")
    b = np.array([0.0, 1.0], dtype="float32")
    results = interpolate_path(library, a, b, top_k=2)
    by_scene = {r.scene_id: r.novel for r in results}
    # Scenes 1 and 3 sit on the anchors themselves.
    assert by_scene.get(1) is False
    assert by_scene.get(3) is False


def test_results_are_deduped_to_one_row_per_scene_per_step(library) -> None:
    """The index carries ~3 keyframes per scene; results are scene-level."""
    _write_film(library, "beta", [[1.0, 0.2], [1.0, 0.2], [1.0, 0.2]])
    a = np.array([1.0, 0.0], dtype="float32")
    b = np.array([0.0, 1.0], dtype="float32")
    results = interpolate_path(library, a, b, steps=(0.5,), top_k=10)
    keys = [(r.film_slug, r.scene_id) for r in results]
    assert len(keys) == len(set(keys))


def test_every_requested_step_is_represented(library) -> None:
    a = np.array([1.0, 0.0], dtype="float32")
    b = np.array([0.0, 1.0], dtype="float32")
    results = interpolate_path(library, a, b, top_k=2)
    assert {r.t for r in results} == set(DEFAULT_STEPS)


def test_slug_filter_restricts_the_search(library) -> None:
    _write_film(library, "beta", [[1.0, 1.0]])
    a = np.array([1.0, 0.0], dtype="float32")
    b = np.array([0.0, 1.0], dtype="float32")
    results = interpolate_path(library, a, b, steps=(0.5,), top_k=10, slugs=["beta"])
    assert {r.film_slug for r in results} == {"beta"}


def test_empty_library_returns_empty(tmp_path) -> None:
    lib = tmp_path / "library"
    lib.mkdir()
    a = np.array([1.0, 0.0], dtype="float32")
    b = np.array([0.0, 1.0], dtype="float32")
    assert interpolate_path(lib, a, b) == []


# ── the novelty diagnostic ───────────────────────────────────────────────────


def test_novelty_rate_reports_the_share_between_the_anchors(library) -> None:
    a = np.array([1.0, 0.0], dtype="float32")
    b = np.array([0.0, 1.0], dtype="float32")
    results = interpolate_path(library, a, b, steps=(0.5,), top_k=1)
    assert novelty_rate(results, 0.5) == 1.0


def test_novelty_rate_of_an_unretrieved_step_is_zero(library) -> None:
    a = np.array([1.0, 0.0], dtype="float32")
    b = np.array([0.0, 1.0], dtype="float32")
    results = interpolate_path(library, a, b, steps=(0.5,), top_k=1)
    assert novelty_rate(results, 0.9) == 0.0
