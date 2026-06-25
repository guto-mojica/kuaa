"""Unit tests for :mod:`kuaa.search.aggregate` — cross-film search.

Targets the new module surface directly (``from kuaa.search.aggregate
import aggregate_search``) so the public entry point of the relocated
function is exercised. The existing per-film and hybrid behaviours are
covered exhaustively by ``test_multi_film_search.py`` (11) and
``test_aggregate_search_hybrid.py`` (7); these three tests focus on the
T11-extracted module's *callable shape*:

  1. text-mode runs end-to-end against a 2-film hermetic library;
  2. non-text modalities raise NotImplementedError (the explicit
     contract bound the function carries while plans 3-5 are deferred);
  3. an empty library short-circuits with ``[]`` and never touches the
     CLIP embedder factory.

Fixture style mirrors ``test_multi_film_search.py`` — inline per-film
JSON + .npy + monkeypatched ``_get_embedder`` on ``kuaa.search.aggregate``
(the canonical home after T13; ``api.services.search`` re-exports the name
but tests must patch the module where the function is actually called).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import kuaa.search.aggregate as _csa_module_ref  # noqa: F401 — ensure loaded
from kuaa.library import register_film
from kuaa.search.aggregate import aggregate_search

# The submodule is shadowed in kuaa.search by the `aggregate` function
# re-export; access via sys.modules to reach the module object reliably.
_AGGREGATE_MODULE = sys.modules["kuaa.search.aggregate"]


def _make_film_with_embeddings(library_dir: Path, slug: str, vectors: list[list[float]]) -> None:
    """Create a minimal per-film layout with a CLIP index.

    Mirrors :func:`tests.test_multi_film_search._make_film_with_embeddings`
    but inlined to keep this file self-contained (the cross-test fixture
    would couple this file to ``test_multi_film_search`` without buying
    much — the helper is 20 lines).
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


def _cfg(library_dir: Path) -> object:
    return SimpleNamespace(paths=SimpleNamespace(library_dir=str(library_dir)))


@pytest.fixture()
def small_library_aggregate_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """2-film hermetic library + stubbed CLIP encoder.

    Both films get a 2-D embedding geometry so the stubbed ``[1, 0]``
    query encodes to the same top-scoring direction in both — the test
    just needs ``aggregate_search`` to walk the library and produce
    results, not to assert on the score arithmetic (that's
    test_multi_film_search's job).
    """
    library_dir = tmp_path / "library"
    library_dir.mkdir()
    register_film(library_dir, slug="a", title="A", year=2000, raw_filename="a.mp4")
    register_film(library_dir, slug="b", title="B", year=2001, raw_filename="b.mp4")
    _make_film_with_embeddings(library_dir, "a", [[1.0, 0.0], [0.0, 1.0]])
    _make_film_with_embeddings(library_dir, "b", [[1.0, 0.0]])

    class StubEmbedder:
        def encode_text(self, q: str) -> np.ndarray:
            return np.array([1.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(_AGGREGATE_MODULE, "_get_embedder", lambda cfg: StubEmbedder())
    return _cfg(library_dir)


def test_aggregate_search_text_mode_runs(small_library_aggregate_fixture: object) -> None:
    """``aggregate_search`` returns hit dicts shaped for the route layer."""
    cfg = small_library_aggregate_fixture
    hits = aggregate_search(
        cfg,
        query="horse",
        modality="text",
        top_k=5,
        retriever_mode="clip",
    )
    assert isinstance(hits, list)
    assert hits, "expected at least one hit from the 2-film hermetic library"
    # Hit dicts carry the cross-film keys the route relies on.
    for h in hits:
        assert {"film_slug", "film_title", "scene_id", "score", "keyframe_path", "timecode"} <= set(
            h
        )


def test_aggregate_search_rejects_non_text_modality(
    small_library_aggregate_fixture: object,
) -> None:
    """Image / audio / fusion modalities raise NotImplementedError (plan 3-5)."""
    cfg = small_library_aggregate_fixture
    with pytest.raises(NotImplementedError, match="modality"):
        aggregate_search(cfg, query="x", modality="image", top_k=1)


def test_aggregate_search_empty_library_short_circuits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Zero registered films → empty list, no CLIP model load.

    The empty-library short-circuit is the same hot path
    ``test_multi_film_search.test_aggregate_search_empty_library_does_not_load_embedder``
    pins for the legacy entry point; this test re-pins it against the
    relocated module to catch any regression in the early-return order
    if the function is later refactored to load the embedder first.
    """
    library_dir = tmp_path / "library"
    library_dir.mkdir()

    def _should_not_load(cfg: object) -> object:
        raise AssertionError("_get_embedder was called on an empty library — eager-load regression")

    monkeypatch.setattr(_AGGREGATE_MODULE, "_get_embedder", _should_not_load)

    hits = aggregate_search(_cfg(library_dir), query="anything", modality="text", top_k=10)
    assert hits == []
