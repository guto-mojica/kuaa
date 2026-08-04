"""The legacy per-film route fuses the same metadata signal as ``find()``.

``f88db82`` wired the lexical metadata list (tags / descriptions / detected
objects) into ``find(mode="hybrid")``, but the older route path —
``api.services.search.dispatch_text_search`` — kept calling ``search_hybrid``
without it, so the same query ranked differently depending on which route
served it. This file pins the parity: both hybrid paths fuse three lists.

Fixture mirrors ``tests/test_find_metadata_signal.py``: scene 2's only
evidence for the query is a YOLO ``book`` detection, and it is the worst
cosine match, so only the metadata leg can lift it.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import api.services.search as search_service
from kuaa.search.bm25 import bm25_index_for_ctx, clear_bm25_cache
from kuaa.search.cache import IndexStatus, SearchIndex

_SCENES = [1, 2, 3]


@pytest.fixture()
def ctx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
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
    monkeypatch.setattr(search_service, "load_index", lambda *a, **k: idx)
    # Skip the api.deps config resolution — build the index straight from ctx.
    monkeypatch.setattr(
        search_service,
        "_get_bm25_index_for_ctx",
        lambda c: bm25_index_for_ctx(c, stopwords_lang=None, k1=1.5, b=0.75),
    )
    return SimpleNamespace(slug="a", metadata_dir=md, embeddings_dir=tmp_path / "embeddings")


def _ranked(ctx: SimpleNamespace, query: str, retriever: str = "hybrid") -> list[int]:
    cfg = SimpleNamespace(
        embeddings=SimpleNamespace(mapping_filename="index_mapping.json", filename="emb.npy")
    )
    df, no_index = search_service.dispatch_text_search(
        cfg,
        ctx,
        query,
        tags=[],
        top_k=3,
        min_sim=0.0,
        retriever=retriever,
        sw=0.5,
        bw=0.5,
        rrf_k=60,
    )
    assert not no_index
    return [int(sid) for sid in df["scene_id"].tolist()]


def test_legacy_hybrid_surfaces_an_object_only_match(ctx: SimpleNamespace) -> None:
    """``book`` appears only in the detector output for scene 2."""
    assert _ranked(ctx, "book")[0] == 2


def test_legacy_portuguese_query_reaches_the_english_detector_class(ctx: SimpleNamespace) -> None:
    assert _ranked(ctx, "livro")[0] == 2


def test_legacy_clip_mode_still_ignores_the_metadata_signal(ctx: SimpleNamespace) -> None:
    assert _ranked(ctx, "book", retriever="clip")[0] == 1
