"""C1 — golden snapshot of aggregate_search. Byte-identical across the decomposition.

Drives the cross-film ``aggregate_search`` (the global-RRF function in
``kuaa.search.aggregate``) over a hermetic 2-film fixture across all
three retriever modes and pins the returned hit list to a golden file. The
snapshot is recorded from the CURRENT (pre-C1) code and MUST stay
byte-identical through every extraction step — any drift means a unit's
behaviour changed.

Fixture mirrors ``tests/test_search_aggregate.py`` (inline per-film
JSON + .npy + monkeypatched ``_get_embedder``) and additionally seeds
``scene_descriptions.json`` / ``scene_tags.json`` so the hybrid metadata
signal is exercised, not just the CLIP/BM25 retrieval lists.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import kuaa.search.aggregate as _agg  # noqa: F401 — ensure loaded
from kuaa.library import register_film
from kuaa.search.aggregate import aggregate_search
from tests._snapshot import assert_snapshot

_AGG_MOD = sys.modules["kuaa.search.aggregate"]


def _make_film(library_dir: Path, slug: str, vectors: list[list[float]]) -> None:
    md = library_dir / slug / "metadata"
    md.mkdir(parents=True)
    emb_dir = library_dir / slug / "embeddings"
    emb_dir.mkdir(parents=True)
    (library_dir / slug / "frames" / "keyframes").mkdir(parents=True)
    arr = np.array(vectors, dtype=np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)
    np.save(emb_dir / "keyframe_embeddings.npy", arr)
    kf_paths = [f"data/library/{slug}/frames/keyframes/{i}.jpg" for i in range(len(vectors))]
    (emb_dir / "index_mapping.json").write_text(
        json.dumps(
            {
                "total_vectors": len(vectors),
                "keyframe_paths": kf_paths,
                "scene_ids": list(range(len(vectors))),
                "keyframe_ids": list(range(len(vectors))),
            }
        )
    )
    (md / "keyframes_metadata.json").write_text(
        json.dumps(
            [
                {"scene_id": i, "filepath": kf_paths[i], "start_time_s": float(i)}
                for i in range(len(vectors))
            ]
        )
    )
    (md / "scene_descriptions.json").write_text(
        json.dumps([{"scene_id": 0, "description": "a man on a horse"}])
    )
    (md / "scene_tags.json").write_text(json.dumps({"outdoor": [0]}))
    (md / "manual_annotations.json").write_text("{}")


@pytest.fixture()
def library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    library_dir = tmp_path / "library"
    library_dir.mkdir()
    register_film(library_dir, slug="a", title="A", year=2000, raw_filename="a.mp4")
    register_film(library_dir, slug="b", title="B", year=2001, raw_filename="b.mp4")
    _make_film(library_dir, "a", [[1.0, 0.0], [0.0, 1.0]])
    _make_film(library_dir, "b", [[1.0, 0.0]])

    class StubEmbedder:
        def encode_text(self, q: str) -> np.ndarray:
            return np.array([1.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(_AGG_MOD, "_get_embedder", lambda cfg: StubEmbedder())
    return SimpleNamespace(paths=SimpleNamespace(library_dir=str(library_dir)))


@pytest.mark.parametrize("mode", ["clip", "bm25", "hybrid"])
def test_aggregate_search_snapshot(library: SimpleNamespace, mode: str) -> None:
    hits = aggregate_search(library, query="horse", modality="text", top_k=5, retriever_mode=mode)
    assert_snapshot(f"aggregate_{mode}", hits)
