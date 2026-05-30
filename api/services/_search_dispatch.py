"""Audio + fusion dispatchers (split from api/services/search.py — Task A1).

Re-exported on ``api.services.search`` so route import paths are unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from api.services.catalog import keyframe_url
from cinemateca.models.base import AudioEmbedder


def dispatch_audio_search(
    cfg: Any,
    ctx: Any | None,
    q: str,
    top_k: int,
) -> tuple[list[dict], bool]:
    """Run CLAP search; return ``(hits, no_index)``.

    ``ctx`` given → per-film; ``ctx=None`` → cross-film aggregate.
    ``no_index=True`` when no CLAP index exists. Embedder loaded at most once.
    Test seam: monkeypatch ``cinemateca.models.registry.get_audio_embedder``.
    """
    from cinemateca.library import scan_library
    from cinemateca.models.registry import get_audio_embedder
    from cinemateca.search.audio import load_audio_index, search_audio

    if ctx is not None:
        audio_dir = Path(ctx.metadata_dir).parent / "audio"
        index = load_audio_index(audio_dir)
        if index is None:
            return [], True
        embedder = get_audio_embedder(cfg, device=None)
        hits = search_audio(index, embedder, q, top_k=top_k)
        for h in hits:
            h["film_slug"] = ctx.slug
        return hits, False

    # Aggregate: walk registry, skip films without a CLAP index.
    library_dir = Path(cfg.paths.library_dir)
    films = list(scan_library(library_dir))
    if not films:
        return [], True
    embedder_agg: AudioEmbedder | None = None
    all_hits: list[dict] = []
    any_index = False
    for film in films:
        film_audio_dir = library_dir / film.slug / "audio"
        idx = load_audio_index(film_audio_dir)
        if idx is None:
            continue
        any_index = True
        if embedder_agg is None:
            embedder_agg = get_audio_embedder(cfg, device=None)
        film_hits = search_audio(idx, embedder_agg, q, top_k=top_k)
        for h in film_hits:
            h["film_slug"] = film.slug
            h["film_title"] = film.title
            all_hits.append(h)
    if not any_index:
        return [], True
    all_hits.sort(key=lambda r: r["score"], reverse=True)
    return all_hits[:top_k], False


class _NullEncoder:
    """Stub encoder for an absent modality in fusion (zero-row stub pattern)."""

    def encode_text(self, text: str) -> Any:  # pragma: no cover - trivial
        import numpy as np

        return np.zeros(1, dtype="float32")


def _normalise_clip_mapping(raw: Any) -> list[dict]:
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


def _fusion_per_film_by_paths(
    *,
    cfg: Any,
    slug: str,
    embeddings_dir: Path,
    audio_dir: Path,
    q: str,
    top_k: int,
    visual_weight: float,
    k_each: int,
    clip_embedder: Any | None,
    clap_embedder: Any | None,
) -> tuple[list[dict], bool, Any | None, Any | None]:
    """Run CLIP×CLAP fusion for one film by paths. Returns ``(hits, no_index, clip_emb, clap_emb)``."""
    import json as _json

    import numpy as np

    from cinemateca.models.registry import get_audio_embedder, get_image_embedder
    from cinemateca.search.audio import load_audio_index
    from cinemateca.search.fusion import FusionConfig, search_fusion

    clip_emb_path = embeddings_dir / "keyframe_embeddings.npy"
    clip_map_path = embeddings_dir / "index_mapping.json"
    has_clip = clip_emb_path.exists() and clip_map_path.exists()
    audio_idx = load_audio_index(audio_dir)
    has_clap = audio_idx is not None

    if not has_clip and not has_clap:
        return [], True, clip_embedder, clap_embedder

    if has_clip and clip_embedder is None:
        clip_embedder = get_image_embedder(cfg, device=None)
    if has_clap and clap_embedder is None:
        clap_embedder = get_audio_embedder(cfg, device=None)

    if has_clip:
        clip_emb = np.load(clip_emb_path).astype("float32", copy=False)
        clip_mapping = _normalise_clip_mapping(_json.loads(clip_map_path.read_text()))
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

    hits = search_fusion(
        clip_emb=clip_emb,
        clap_emb=clap_emb,
        clip_mapping=clip_mapping,
        clap_mapping=clap_mapping,
        query_text=q,
        clip_embedder=cast(Any, clip_embedder if has_clip else _NullEncoder()),
        clap_embedder=cast(Any, clap_embedder if has_clap else _NullEncoder()),
        cfg=FusionConfig(visual_weight=visual_weight, k_each=k_each, k_final=top_k),
    )
    for h in hits:
        h["film_slug"] = slug
    return hits, False, clip_embedder, clap_embedder


def dispatch_fusion_search(
    cfg: Any,
    ctx: Any | None,
    q: str,
    top_k: int,
    *,
    visual_weight: float = 0.5,
    k_each: int = 50,
) -> tuple[list[dict], bool]:
    """CLIP×CLAP fusion search; returns ``(hits, no_index)``.

    ``ctx`` given → per-film; ``ctx=None`` → cross-film aggregate.
    Missing modalities contribute zero rows. Aggregate path derives paths
    directly from ``cfg.paths.library_dir`` to avoid ``FilmContext.for_film``
    side-effects (mkdir). Test seam: monkeypatch registry embedder factories.
    """
    from cinemateca.library import scan_library

    if ctx is not None:
        hits, no_index, _, _ = _fusion_per_film_by_paths(
            cfg=cfg,
            slug=ctx.slug,
            embeddings_dir=ctx.embeddings_dir,
            audio_dir=Path(ctx.metadata_dir).parent / "audio",
            q=q,
            top_k=top_k,
            visual_weight=visual_weight,
            k_each=k_each,
            clip_embedder=None,
            clap_embedder=None,
        )
        return hits, no_index

    library_dir = Path(cfg.paths.library_dir)
    films = list(scan_library(library_dir))
    if not films:
        return [], True
    clip_embedder: Any | None = None
    clap_embedder: Any | None = None
    all_hits: list[dict] = []
    any_film = False
    for film in films:
        film_hits, film_no_index, clip_embedder, clap_embedder = _fusion_per_film_by_paths(
            cfg=cfg,
            slug=film.slug,
            embeddings_dir=library_dir / film.slug / "embeddings",
            audio_dir=library_dir / film.slug / "audio",
            q=q,
            top_k=top_k,
            visual_weight=visual_weight,
            k_each=k_each,
            clip_embedder=clip_embedder,
            clap_embedder=clap_embedder,
        )
        if film_no_index:
            continue
        any_film = True
        for h in film_hits:
            h["film_title"] = film.title
            all_hits.append(h)
    if not any_film:
        return [], True
    all_hits.sort(key=lambda r: r["score"], reverse=True)
    return all_hits[:top_k], False


def audio_hits_to_template_dicts(
    cfg: Any, hits: list[dict], *, per_film_slug: str | None = None
) -> list[dict]:
    """Convert CLAP hits to template-card dicts (score→similarity, resolves keyframe + timecode)."""
    from cinemateca.library import FilmContext, derive_fps, load_json, to_smpte

    data_dir = Path(cfg.paths.data_dir).resolve()
    kf_cache: dict[str, tuple[dict, float]] = {}

    def _kf_for(slug: str) -> tuple[dict, float]:
        if slug in kf_cache:
            return kf_cache[slug]
        try:
            ctx = FilmContext.for_film(cfg, slug)
        except ValueError:
            kf_cache[slug] = ({}, 24.0)
            return kf_cache[slug]
        raw_kf = load_json(ctx.metadata_dir / "keyframes_metadata.json")
        kf_meta: list[Any] = raw_kf if isinstance(raw_kf, list) else []
        by_scene = {int(e["scene_id"]): e for e in kf_meta if "scene_id" in e}
        kf_cache[slug] = (by_scene, derive_fps(kf_meta))
        return kf_cache[slug]

    out: list[dict] = []
    for h in hits:
        slug = h.get("film_slug") or per_film_slug or ""
        sid = int(h["scene_id"])
        by_scene, fps = _kf_for(slug) if slug else ({}, 24.0)
        meta = by_scene.get(sid) or {}
        kf_path = meta.get("filepath", "") or meta.get("keyframe_path", "") or ""
        start_s = float(meta.get("start_time_s") or 0.0)
        out.append(
            {
                "film_slug": slug,
                "scene_id": sid,
                "similarity": float(h["score"]),
                "img_url": keyframe_url(kf_path, data_dir) if kf_path else None,
                "timecode": to_smpte(start_s, fps) if start_s > 0 else "",
            }
        )
    return out
