"""Cross-modal CLIP × CLAP fusion search (M3).

Linear late-fusion: ``score = w * clip_cosine + (1 - w) * clap_cosine``,
where w is ``cfg.visual_weight``. Both embedding spaces are
L2-normalised at write time, so cosine reduces to a dot product.

**No alternative fusion algorithms** — RRF / score-rank / learned fusion
are explicitly out of scope per the M3 spec freeze line.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FusionConfig:
    visual_weight: float = 0.5
    k_each: int = 50      # per-modality top-k pulled before merge
    k_final: int = 10     # final returned length


class _TextEncoder(Protocol):
    def encode_text(self, text: str) -> np.ndarray: ...


def search_fusion(
    *,
    clip_emb: np.ndarray,
    clap_emb: np.ndarray,
    clip_mapping: list[dict],
    clap_mapping: list[dict],
    query_text: str,
    clip_embedder: _TextEncoder,
    clap_embedder: _TextEncoder,
    cfg: FusionConfig,
) -> list[dict]:
    """Run linear-late-fusion CLIP × CLAP retrieval for one film.

    Args:
        clip_emb: (N_clip, D_clip) L2-normalised CLIP keyframe embeddings.
        clap_emb: (N_clap, D_clap) L2-normalised CLAP scene embeddings.
        clip_mapping: parallel list ``[{"scene_id": int, ...}, …]`` of length N_clip.
        clap_mapping: same shape, length N_clap (may be shorter than CLIP
            when audio extraction is incomplete or disabled).
        query_text: free-text query — same string passed to both encoders.
        clip_embedder: any object with ``encode_text(str) -> (D_clip,)``.
            In production this is the existing CLIP backend.
        clap_embedder: ``encode_text(str) -> (D_clap,)``. Production: CLAP backend.
        cfg: ``FusionConfig`` — ``visual_weight``, ``k_each``, ``k_final``.

    Returns:
        ``[{"scene_id": int, "score": float, "clip_score": float,
        "clap_score": float}, …]`` sorted descending by ``score``,
        length ``min(k_final, |union|)``.

    Semantics:
        Scenes with only CLIP coverage contribute ``w * clip_cosine``;
        scenes with only CLAP coverage contribute ``(1-w) * clap_cosine``.
        Both-sides scenes get the linear combine. This is the simplest
        defensible behaviour: missing modalities should not actively
        penalise; they just don't add their term.
    """
    w = float(cfg.visual_weight)
    if not 0.0 <= w <= 1.0:
        raise ValueError(f"visual_weight must be in [0, 1], got {w}")
    if not query_text.strip():
        return []

    q_clip = clip_embedder.encode_text(query_text)
    q_clap = clap_embedder.encode_text(query_text)
    if q_clip.shape[0] != clip_emb.shape[1]:
        raise ValueError(
            f"CLIP query dim {q_clip.shape[0]} vs index dim {clip_emb.shape[1]}"
        )
    if q_clap.shape[0] != clap_emb.shape[1]:
        raise ValueError(
            f"CLAP query dim {q_clap.shape[0]} vs index dim {clap_emb.shape[1]}"
        )

    clip_cos = clip_emb @ q_clip  # (N_clip,)
    clap_cos = clap_emb @ q_clap  # (N_clap,)

    # Per-modality top-k_each, then merge — bounds the cross product when
    # libraries grow (a Jeca-Tatu-scale film has ~400 scenes; merging is
    # trivial there, but the bound matters when N grows to 10k+).
    clip_scores: dict[int, float] = {}
    if clip_emb.shape[0] > 0:
        k_clip = min(int(cfg.k_each), clip_emb.shape[0])
        top_clip = np.argpartition(-clip_cos, k_clip - 1)[:k_clip]
        for i in top_clip:
            sid = int(clip_mapping[int(i)]["scene_id"])
            clip_scores[sid] = float(clip_cos[int(i)])

    clap_scores: dict[int, float] = {}
    if clap_emb.shape[0] > 0:
        k_clap = min(int(cfg.k_each), clap_emb.shape[0])
        top_clap = np.argpartition(-clap_cos, k_clap - 1)[:k_clap]
        for i in top_clap:
            sid = int(clap_mapping[int(i)]["scene_id"])
            clap_scores[sid] = float(clap_cos[int(i)])

    all_sids = set(clip_scores) | set(clap_scores)
    rows: list[dict] = []
    for sid in all_sids:
        cs = clip_scores.get(sid, 0.0)
        as_ = clap_scores.get(sid, 0.0)
        rows.append({
            "scene_id": sid,
            "score": w * cs + (1.0 - w) * as_,
            "clip_score": cs,
            "clap_score": as_,
        })
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows[: int(cfg.k_final)]
