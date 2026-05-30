"""CLAP / fusion hit-to-template-dict conversion (split from _search_dispatch — G1 fix).

Re-exported on ``api.services.search`` and ``api.services._search_dispatch``
so all existing import paths remain valid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def audio_hits_to_template_dicts(
    cfg: Any, hits: list[dict], *, per_film_slug: str | None = None
) -> list[dict]:
    """Convert CLAP hits to template-card dicts (score→similarity, resolves keyframe + timecode)."""
    from api.services.catalog import keyframe_url
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
