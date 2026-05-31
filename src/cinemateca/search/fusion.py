"""Cross-modal CLIP × CLAP fusion search (M3).

Linear late-fusion: ``score = w * clip_cosine + (1 - w) * clap_cosine``,
where w is ``cfg.visual_weight``. Both embedding spaces are
L2-normalised at write time, so cosine reduces to a dot product.

**No alternative fusion algorithms** — RRF / score-rank / learned fusion
are explicitly out of scope per the M3 spec freeze line.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

from cinemateca.config import Settings
from cinemateca.search.audio import load_audio_index
from cinemateca.search.types import Hit, Query, SearchMode, SearchResult
from cinemateca.timing import timed


@dataclass(frozen=True)
class FusionConfig:
    visual_weight: float = 0.5
    k_each: int = 50  # per-modality top-k pulled before merge
    k_final: int = 10  # final returned length


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
    if q_clip.ndim != 1 or q_clip.shape[0] != clip_emb.shape[1]:
        raise ValueError(
            f"CLIP query vector dim {q_clip.shape} incompatible with index dim {clip_emb.shape[1]}"
        )
    if q_clap.ndim != 1 or q_clap.shape[0] != clap_emb.shape[1]:
        raise ValueError(
            f"CLAP query vector dim {q_clap.shape} incompatible with index dim {clap_emb.shape[1]}"
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
        rows.append(
            {
                "scene_id": sid,
                "score": w * cs + (1.0 - w) * as_,
                "clip_score": cs,
                "clap_score": as_,
            }
        )
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows[: int(cfg.k_final)]


# ── Typed fusion verbs (C9) ───────────────────────────────────────────────────
# ``find_fusion`` / ``aggregate_fusion`` are the fusion analogue of
# ``cinemateca.search.find`` / ``aggregate``: they own the per-film CLIP + CLAP
# index loading, wrap the ``search_fusion`` leaf, time the search, and return a
# typed :class:`SearchResult` carrying the 5 per-query metadata fields
# (``fusion_used=True``, ``retriever_mode="fusion"``, ``reranker_applied=False``,
# ``num_films_searched``, ``latency_ms``). That typed metadata is plumbing for
# future eval grouping / UI affordances — it is *not* consumed by
# ``cinemateca.eval`` today: eval calls the ``search_fusion`` leaf directly (see
# ``cinemateca.eval.slates``). The leaf stays ``list[dict]`` precisely so eval
# keeps consuming it unchanged. The api dispatcher projects
# ``SearchResult.hits`` back to template view-dicts, preserving the HTTP shape.
#
# The loading helpers (``normalise_clip_mapping`` / ``NullEncoder``) are the
# canonical core copies; the api dispatcher delegates here rather than keeping
# its own duplicates. ``mode`` is the benign ``"clip"`` placeholder (see the
# audio module note); ``retriever_mode="fusion"`` is the C9 semantic field.
_FUSION_PLACEHOLDER_MODE: SearchMode = "clip"


class NullEncoder:
    """Stub encoder for an absent modality in fusion (zero-row stub pattern)."""

    def encode_text(self, text: str) -> np.ndarray:  # pragma: no cover - trivial
        return np.zeros(1, dtype="float32")


def normalise_clip_mapping(raw: Any) -> list[dict]:
    """Coerce ``index_mapping.json`` to ``list[dict]`` with ``scene_id`` keys."""
    if isinstance(raw, dict) and "scene_ids" in raw:
        sids = raw["scene_ids"]
        return [{"scene_id": int(sids[i])} for i in range(len(sids))]
    if isinstance(raw, list):
        return [{"scene_id": int(m["scene_id"])} for m in raw]
    raise ValueError(
        "Unrecognised CLIP index_mapping shape: expected dict with "
        "'scene_ids' key or list of dicts."
    )


def _fusion_rows_for_film(
    *,
    cfg: Settings,
    embeddings_dir: Path,
    audio_dir: Path,
    query_text: str,
    top_k: int,
    visual_weight: float,
    k_each: int,
    clip_embedder: Any | None,
    clap_embedder: Any | None,
    image_embedder_factory: Any,
    audio_embedder_factory: Any,
) -> tuple[list[dict], bool, Any | None, Any | None]:
    """Run CLIP×CLAP fusion for one film by paths.

    Returns ``(rows, no_index, clip_embedder, clap_embedder)``. The embedders are
    threaded through so the aggregate path loads each at most once. Missing
    modalities contribute zero rows (a film with only CLIP still ranks by its
    visual term). ``no_index=True`` when neither index exists.
    """
    clip_emb_path = embeddings_dir / "keyframe_embeddings.npy"
    clip_map_path = embeddings_dir / "index_mapping.json"
    has_clip = clip_emb_path.exists() and clip_map_path.exists()
    audio_idx = load_audio_index(audio_dir)
    has_clap = audio_idx is not None

    if not has_clip and not has_clap:
        return [], True, clip_embedder, clap_embedder

    if has_clip and clip_embedder is None:
        clip_embedder = image_embedder_factory(cfg)
    if has_clap and clap_embedder is None:
        clap_embedder = audio_embedder_factory(cfg)

    if has_clip:
        clip_emb = np.load(clip_emb_path).astype("float32", copy=False)
        clip_mapping = normalise_clip_mapping(json.loads(clip_map_path.read_text()))
    else:
        clip_emb = np.zeros((0, 1), dtype="float32")
        clip_mapping = []

    if has_clap:
        assert audio_idx is not None  # narrow for mypy
        clap_emb = audio_idx.embeddings
        clap_mapping = [{"scene_id": int(m["scene_id"])} for m in audio_idx.mapping]
    else:
        clap_emb = np.zeros((0, 1), dtype="float32")
        clap_mapping = []

    rows = search_fusion(
        clip_emb=clip_emb,
        clap_emb=clap_emb,
        clip_mapping=clip_mapping,
        clap_mapping=clap_mapping,
        query_text=query_text,
        clip_embedder=cast(Any, clip_embedder if has_clip else NullEncoder()),
        clap_embedder=cast(Any, clap_embedder if has_clap else NullEncoder()),
        cfg=FusionConfig(visual_weight=visual_weight, k_each=k_each, k_final=top_k),
    )
    return rows, False, clip_embedder, clap_embedder


def _fusion_hit(row: dict, *, film_slug: str | None, film_title: str | None) -> Hit:
    """Lift one ``search_fusion`` row to a typed :class:`Hit` (join keys + score)."""
    return Hit(
        scene_id=int(row["scene_id"]),
        score=float(row["score"]),
        keyframe_path="",
        film_slug=film_slug,
        film_title=film_title,
    )


def find_fusion(
    cfg: Settings,
    *,
    slug: str,
    embeddings_dir: Path,
    audio_dir: Path,
    query_text: str,
    top_k: int = 10,
    visual_weight: float = 0.5,
    k_each: int = 50,
    image_embedder_factory: Any,
    audio_embedder_factory: Any,
) -> SearchResult:
    """Single-film CLIP×CLAP fusion → typed :class:`SearchResult` (``num_films_searched=1``).

    Loads the film's CLIP + CLAP indexes by path (no ``FilmContext.for_film``
    mkdir side-effects), times the fused search, and stamps ``slug`` onto each
    :class:`Hit`. ``no_index=True`` when the film has neither index. The
    embedder factories are ``cfg -> embedder`` callables (the registry
    ``get_image_embedder`` / ``get_audio_embedder``), injected to keep this free
    of a hard registry import.
    """
    with timed("find.fusion") as t:
        rows, no_index, _, _ = _fusion_rows_for_film(
            cfg=cfg,
            embeddings_dir=embeddings_dir,
            audio_dir=audio_dir,
            query_text=query_text,
            top_k=top_k,
            visual_weight=visual_weight,
            k_each=k_each,
            clip_embedder=None,
            clap_embedder=None,
            image_embedder_factory=image_embedder_factory,
            audio_embedder_factory=audio_embedder_factory,
        )
    hits = [_fusion_hit(r, film_slug=slug, film_title=None) for r in rows]
    return SearchResult(
        hits=hits,
        mode=_FUSION_PLACEHOLDER_MODE,
        weights=None,
        query=Query.of_text(query_text),
        no_index=no_index,
        retriever_mode="fusion",
        fusion_used=True,
        reranker_applied=False,
        num_films_searched=0 if no_index else 1,
        latency_ms=t.elapsed_ms,
    )


def aggregate_fusion(
    cfg: Settings,
    query_text: str,
    *,
    top_k: int = 10,
    visual_weight: float = 0.5,
    k_each: int = 50,
    image_embedder_factory: Any,
    audio_embedder_factory: Any,
) -> SearchResult:
    """Cross-film CLIP×CLAP fusion → typed :class:`SearchResult` (``num_films_searched=N``).

    Walks the registry (``cfg.paths.library_dir``), derives per-film paths
    directly (avoiding ``FilmContext.for_film`` side-effects), skips films with
    neither index, runs the fused search per film, then merges + sorts
    descending. ``num_films_searched`` counts films that actually contributed a
    fused list. ``no_index=True`` when no film had either index. Embedders are
    loaded at most once across the walk.
    """
    from cinemateca.library import scan_library

    library_dir = Path(cfg.paths.library_dir)
    films = list(scan_library(library_dir))
    clip_embedder: Any | None = None
    clap_embedder: Any | None = None
    all_hits: list[Hit] = []
    films_searched = 0
    with timed("aggregate.fusion") as t:
        for film in films:
            rows, no_index, clip_embedder, clap_embedder = _fusion_rows_for_film(
                cfg=cfg,
                embeddings_dir=library_dir / film.slug / "embeddings",
                audio_dir=library_dir / film.slug / "audio",
                query_text=query_text,
                top_k=top_k,
                visual_weight=visual_weight,
                k_each=k_each,
                clip_embedder=clip_embedder,
                clap_embedder=clap_embedder,
                image_embedder_factory=image_embedder_factory,
                audio_embedder_factory=audio_embedder_factory,
            )
            if no_index:
                continue
            films_searched += 1
            all_hits.extend(
                _fusion_hit(r, film_slug=film.slug, film_title=film.title) for r in rows
            )
    all_hits.sort(key=lambda h: h.score, reverse=True)
    return SearchResult(
        hits=all_hits[:top_k],
        mode=_FUSION_PLACEHOLDER_MODE,
        weights=None,
        query=Query.of_text(query_text),
        no_index=films_searched == 0,
        retriever_mode="fusion",
        fusion_used=True,
        reranker_applied=False,
        num_films_searched=films_searched,
        latency_ms=t.elapsed_ms,
    )
