"""Single-film ``find(mode="hybrid")`` folds in the lexical metadata signal.

Cross-film ``aggregate`` has always fused a third, exact-match list (tags /
descriptions / detected objects) alongside CLIP and BM25. ``find`` — the verb
behind the main search UI — did not, so a scene whose only evidence was a
detector class could not be retrieved at all: CLIP ranks it on appearance and
BM25 only sees caption text.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import kuaa.search._dispatch as dispatch_mod
from kuaa.search._dispatch import find
from kuaa.search.bm25 import clear_bm25_cache
from kuaa.search.cache import IndexStatus, SearchIndex
from kuaa.search.types import Query

# Scene 2 is the target: a book is on screen, but nothing in its caption or
# tags says so and it looks nothing like the query vector.
_SCENES = [1, 2, 3]


@pytest.fixture()
def film(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    clear_bm25_cache()
    md = tmp_path / "metadata"
    md.mkdir()

    (md / "keyframes_metadata.json").write_text(
        json.dumps(
            [{"scene_id": sid, "filepath": f"/lib/a/scene_{sid:04d}_kf_01.jpg"} for sid in _SCENES]
        )
    )
    (md / "scene_descriptions.json").write_text(
        json.dumps(
            [
                {"scene_id": 1, "keyframe_id": "scene_0001_kf_01", "description": "An empty road."},
                {"scene_id": 2, "keyframe_id": "scene_0002_kf_01", "description": "A quiet room."},
                {"scene_id": 3, "keyframe_id": "scene_0003_kf_01", "description": "A wide field."},
            ]
        )
    )
    (md / "scene_tags.json").write_text(json.dumps({"interior": [2]}))
    (md / "manual_annotations.json").write_text("{}")
    (md / "visual_analysis.json").write_text(
        json.dumps(
            [
                {
                    "frame_path": "scene_0002_kf_01.jpg",
                    "object_detection": {
                        "objects": [{"class": "book"}],
                        "class_counts": {"book": 1},
                    },
                }
            ]
        )
    )

    # Scene 2 is the WORST cosine match, so only the metadata signal can lift it.
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]], dtype=np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    kf_df = pd.DataFrame(
        {
            "scene_id": _SCENES,
            "filepath": [f"/lib/a/scene_{s:04d}_kf_01.jpg" for s in _SCENES],
        }
    )

    class _Emb:
        def encode_text(self, q: str) -> np.ndarray:
            return np.array([1.0, 0.0], dtype=np.float32)

    idx = SearchIndex(IndexStatus.OK, embeddings=embeddings, kf_df=kf_df, embedder=_Emb())
    monkeypatch.setattr(dispatch_mod, "load_index", lambda *a, **k: idx)
    return SimpleNamespace(slug="a", metadata_dir=md, embeddings_dir=tmp_path / "embeddings")


def _ranked(film: SimpleNamespace, query: str, mode: str = "hybrid") -> list[int]:
    result = find(Query.of_text(query), film=film, mode=mode, top_k=3)
    return [h.scene_id for h in result.hits]


def test_hybrid_surfaces_an_object_only_match(film: SimpleNamespace) -> None:
    """``book`` appears only in the detector output for scene 2."""
    assert _ranked(film, "book")[0] == 2


def test_clip_mode_still_ignores_the_metadata_signal(film: SimpleNamespace) -> None:
    """The signal is hybrid-only — ``clip`` mode stays a pure cosine ranking."""
    assert _ranked(film, "book", mode="clip")[0] == 1


def test_portuguese_query_reaches_the_english_detector_class(film: SimpleNamespace) -> None:
    assert _ranked(film, "livro")[0] == 2


def test_long_query_leaves_fusion_two_way(film: SimpleNamespace) -> None:
    """The scorer is short-query-only, so a sentence must not promote scene 2."""
    assert _ranked(film, "a long descriptive sentence about an empty road")[0] != 2


# No tag-only case here on purpose: BM25 already folds tag tokens into the
# corpus (``tag_boost``), so a tag query ranks its scene first with or without
# the metadata leg. Detected objects are the signal that had no other route
# into ``find``. The scorer's own tag/description legs are covered by
# ``tests/test_aggregate_units.py``.
