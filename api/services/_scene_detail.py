"""Scene inspector context builder (single-scene right-pane detail)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from api.services.catalog import (
    derive_fps,
    keyframe_url,
    load_json,
    load_tag_index,
    to_smpte,
)
from cinemateca.library import FilmContext
from api.services.film_service import list_films

logger = logging.getLogger(__name__)

_VALID_TABS = ("activity", "annotations", "properties")
_TIPOS = ("cartela", "dialogo", "exterior", "interior", "transicao")


def tipo_of(tags: list[str], description: str | None) -> str:
    """Classify a scene into one of the Mojica tipo buckets."""
    desc = (description or "").lower()
    if "title" in desc or any(
        "white-writing" in t or "cartela" in t or "title-card" in t for t in tags
    ):
        return "cartela"
    if any("interior" in t or "baixa-luz" in t for t in tags):
        return "interior"
    if "exterior" in tags or any("rural" in t for t in tags):
        return "exterior"
    if any("duas-pessoas" in t or "dialogo" in t for t in tags):
        return "dialogo"
    return "transicao"


def _resolve_tab(tab: str | None) -> str:
    if tab in _VALID_TABS:
        return tab
    return "activity"


def _scene_lookup(kf_meta: list, scene_id: int) -> dict | None:
    for entry in kf_meta:
        try:
            if int(entry.get("scene_id")) == scene_id:
                return entry
        except (TypeError, ValueError):
            continue
    return None


def _films_by_slug(cfg: Any) -> dict:
    library_dir = Path(cfg.paths.library_dir)
    return {film.slug: film for film in list_films(library_dir)}


def _description_for(metadata_dir: Path, scene_id: int) -> str:
    descs = load_json(metadata_dir / "scene_descriptions.json") or []
    if not isinstance(descs, list):
        return ""
    for entry in descs:
        sid = entry.get("scene_id")
        if sid is None:
            continue
        try:
            if int(sid) == scene_id:
                return str(entry.get("description") or "")
        except (TypeError, ValueError):
            continue
    return ""


def _tags_for(metadata_dir: Path, scene_id: int) -> list[str]:
    merged = load_tag_index(metadata_dir) or {}
    tags: list[str] = []
    for tag, sids in merged.items():
        if not isinstance(sids, (list, set, tuple)):
            continue
        for sid in sids:
            try:
                if int(sid) == scene_id:
                    tags.append(tag)
                    break
            except (TypeError, ValueError):
                continue
    return tags


def build_inspector_context(
    cfg: Any,
    *,
    scene_id: int,
    slug: str | None,
    inspector_tab: str = "activity",
) -> dict | None:
    """Build the template context for the right-pane inspector partial.

    Returns None when the (slug, scene_id) pair cannot be resolved.

    Args:
        cfg: Loaded application config.
        scene_id: Integer scene identifier.
        slug: Film slug; None returns None (404).
        inspector_tab: Active tab key; unknown values fall back to 'activity'.

    Returns:
        Context dict with selected_scene, selected_film, inspector_tab, rhymes;
        or None if the scene cannot be resolved.
    """
    tab = _resolve_tab(inspector_tab)

    if not slug:
        logger.info("inspector: no slug → 404 (scene_id=%d)", scene_id)
        return None

    try:
        ctx = FilmContext.for_film(cfg, slug)
    except ValueError as exc:
        logger.info("inspector: unresolvable slug %r → 404 (%s)", slug, exc)
        return None

    kf_meta = load_json(ctx.metadata_dir / "keyframes_metadata.json") or []
    if not isinstance(kf_meta, list):
        kf_meta = []
    entry = _scene_lookup(kf_meta, scene_id)
    if entry is None:
        logger.info("inspector: scene_id=%d not in %s → 404", scene_id, ctx.metadata_dir)
        return None

    films_by_slug = _films_by_slug(cfg)
    selected_film = films_by_slug.get(slug)
    if selected_film is None:
        logger.info("inspector: slug %r resolves on disk but is not in films.json", slug)

    fps = derive_fps(kf_meta)
    start_s = float(entry.get("start_time_s") or 0.0)
    timecode = to_smpte(start_s, fps) if start_s > 0 else ""
    end_s = float(entry.get("end_time_s") or 0.0)
    duration_s = max(0.0, end_s - start_s)

    img_url = keyframe_url(entry.get("filepath", ""), ctx.data_dir)

    description = _description_for(ctx.metadata_dir, scene_id)
    tags = _tags_for(ctx.metadata_dir, scene_id)

    total_scenes = len(kf_meta)
    scene_index = 1
    for i, e in enumerate(kf_meta, start=1):
        try:
            if int(e.get("scene_id")) == scene_id:
                scene_index = i
                break
        except (TypeError, ValueError):
            continue

    tipo = tipo_of(tags, description)

    selected_scene = {
        "id": scene_id,
        "scene_id": scene_id,
        "film_slug": slug,
        "keyframe_url": img_url or "",
        "timecode": timecode,
        "start_s": start_s,
        "end_s": end_s,
        "duration_s": duration_s,
        "tipo": tipo,
        "title": None,
        "description": description,
        "tags": tags,
        "pin_count": 0,
        "activity_count": 0,
        "annotation_count": 0,
        "pin": None,
        "signals": None,
        "described_when": "",
        "scene_index": scene_index,
        "scene_total": total_scenes,
    }

    return {
        "selected_scene": selected_scene,
        "selected_film": selected_film,
        "inspector_tab": tab,
        "rhymes": [],
    }
