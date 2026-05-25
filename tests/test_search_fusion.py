"""Fusion search tests.

Linear late-fusion: score = w * clip_cosine + (1-w) * clap_cosine.
At w=1.0 must equal pure CLIP retrieval. At w=0.0 must equal pure CLAP.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cinemateca.search.fusion import FusionConfig, search_fusion


def _build_index_pair(tmp_path: Path, n: int = 6, dim: int = 4):
    """Write a CLIP-style + CLAP-style index under tmp_path."""
    rng = np.random.default_rng(0)
    clip = rng.standard_normal((n, dim)).astype("float32")
    clip /= np.linalg.norm(clip, axis=1, keepdims=True)
    clap = rng.standard_normal((n, dim)).astype("float32")
    clap /= np.linalg.norm(clap, axis=1, keepdims=True)
    return clip, clap


class _StubClipEmbedder:
    """Returns ``encode_text`` as a known row, so cosine = 1.0 for that row."""
    def __init__(self, vec: np.ndarray) -> None:
        self.vec = vec
    def encode_text(self, text: str) -> np.ndarray:
        return self.vec.copy()


class _StubClapEmbedder:
    def __init__(self, vec: np.ndarray) -> None:
        self.vec = vec
    def encode_text(self, text: str) -> np.ndarray:
        return self.vec.copy()


def test_fusion_w_one_reduces_to_pure_clip(tmp_path: Path) -> None:
    clip, clap = _build_index_pair(tmp_path, n=6, dim=4)
    # Stub query embeddings: query equals clip-row-2 → CLIP top-1 is sid=2.
    clip_stub = _StubClipEmbedder(clip[2])
    clap_stub = _StubClapEmbedder(np.zeros(4, dtype="float32"))  # zero CLAP score
    hits = search_fusion(
        clip_emb=clip,
        clap_emb=clap,
        clip_mapping=[{"scene_id": i} for i in range(6)],
        clap_mapping=[{"scene_id": i} for i in range(6)],
        query_text="q",
        clip_embedder=clip_stub,
        clap_embedder=clap_stub,
        cfg=FusionConfig(visual_weight=1.0, k_each=6, k_final=3),
    )
    assert hits[0]["scene_id"] == 2
    assert hits[0]["score"] == pytest.approx(1.0, abs=1e-5)


def test_fusion_w_zero_reduces_to_pure_clap(tmp_path: Path) -> None:
    clip, clap = _build_index_pair(tmp_path, n=6, dim=4)
    clip_stub = _StubClipEmbedder(np.zeros(4, dtype="float32"))
    clap_stub = _StubClapEmbedder(clap[4])
    hits = search_fusion(
        clip_emb=clip,
        clap_emb=clap,
        clip_mapping=[{"scene_id": i} for i in range(6)],
        clap_mapping=[{"scene_id": i} for i in range(6)],
        query_text="q",
        clip_embedder=clip_stub,
        clap_embedder=clap_stub,
        cfg=FusionConfig(visual_weight=0.0, k_each=6, k_final=3),
    )
    assert hits[0]["scene_id"] == 4
    assert hits[0]["score"] == pytest.approx(1.0, abs=1e-5)


def test_fusion_default_weight_linear_combine_matches_manual_numpy() -> None:
    clip = np.eye(4, dtype="float32")  # rows are basis vectors
    clap = np.flipud(np.eye(4, dtype="float32"))
    # Query encoded as basis-row-0 for CLIP, basis-row-0 for CLAP.
    # CLIP scores: [1, 0, 0, 0]
    # CLAP scores: [0, 0, 0, 1]  (because clap[3] == [1,0,0,0])
    # Combined w=0.5: [0.5, 0, 0, 0.5] — tie between sid=0 and sid=3.
    clip_stub = _StubClipEmbedder(clip[0])
    clap_stub = _StubClapEmbedder(clap[3])
    hits = search_fusion(
        clip_emb=clip,
        clap_emb=clap,
        clip_mapping=[{"scene_id": i} for i in range(4)],
        clap_mapping=[{"scene_id": i} for i in range(4)],
        query_text="q",
        clip_embedder=clip_stub,
        clap_embedder=clap_stub,
        cfg=FusionConfig(visual_weight=0.5, k_each=4, k_final=4),
    )
    scores = {h["scene_id"]: h["score"] for h in hits}
    assert scores[0] == pytest.approx(0.5)
    assert scores[3] == pytest.approx(0.5)
    assert scores[1] == pytest.approx(0.0)
    assert scores[2] == pytest.approx(0.0)


def test_fusion_handles_missing_scene_in_one_modality() -> None:
    """A scene present in CLIP but not in CLAP (audio not extracted yet)
    should still rank by its CLIP-only contribution at w * clip_score."""
    clip = np.eye(3, dtype="float32")
    clap = np.eye(3, dtype="float32")[:2]  # only 2 audio rows
    clip_stub = _StubClipEmbedder(clip[2])  # match scene 2 (no audio)
    clap_stub = _StubClapEmbedder(np.zeros(3, dtype="float32"))
    hits = search_fusion(
        clip_emb=clip,
        clap_emb=clap,
        clip_mapping=[{"scene_id": i} for i in range(3)],
        clap_mapping=[{"scene_id": 0}, {"scene_id": 1}],  # no sid=2
        query_text="q",
        clip_embedder=clip_stub,
        clap_embedder=clap_stub,
        cfg=FusionConfig(visual_weight=0.7, k_each=3, k_final=3),
    )
    # sid=2 → 0.7 * 1.0 + 0.3 * 0.0 = 0.7 (top); sid=0/1 → 0.
    assert hits[0]["scene_id"] == 2
    assert hits[0]["score"] == pytest.approx(0.7, abs=1e-5)
