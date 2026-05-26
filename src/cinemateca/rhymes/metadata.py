"""Per-scene metadata loaders for rhyme enrichment.

Loads description text, tag list, timecode for a single scene. These
support ``enrich_rhyme`` (in ``api/services/rhymes_service.py``) which
decorates raw :class:`~cinemateca.rhymes.Rhyme` dataclass instances with
the human-readable bits the template needs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from cinemateca.library import (
    FilmContext,
    derive_fps,
    keyframe_url,
    load_json,
    load_tag_index,
    to_smpte,
)

logger = logging.getLogger(__name__)


def load_scene_meta(cfg: Any, slug: str, scene_id: int) -> dict | None:
    """Return the anchor scene's metadata dict, or ``None`` if unresolvable.

    Mirrors :func:`api.services.scenes_service.build_inspector_context`'s
    on-disk lookup pattern: read ``keyframes_metadata.json`` for the
    keyframe + timecode, ``scene_descriptions.json`` for the moondream
    caption, and ``scene_tags.json`` / ``manual_annotations.json``
    (merged) for the tag list. Anything the file system cannot answer
    collapses to a sensible default (``""`` / ``[]``) so the template
    never sees ``None`` on a sub-field.

    Returns ``None`` only when the scene id itself cannot be located in
    the film's keyframe metadata — that is the signal Task 22's template
    uses to render the "anchor missing" empty state.
    """
    try:
        ctx = FilmContext.for_film(cfg, slug)
    except ValueError as exc:
        logger.info("rimas: unresolvable slug %r → empty anchor (%s)", slug, exc)
        return None

    kf_meta = load_json(ctx.metadata_dir / "keyframes_metadata.json") or []
    if not isinstance(kf_meta, list):
        return None

    entry: dict | None = None
    for e in kf_meta:
        try:
            if int(e.get("scene_id")) == scene_id:
                entry = e
                break
        except (TypeError, ValueError):
            continue
    if entry is None:
        return None

    fps = derive_fps(kf_meta)
    start_s = float(entry.get("start_time_s") or 0.0)
    end_s = float(entry.get("end_time_s") or 0.0)
    timecode = to_smpte(start_s, fps) if start_s > 0 else ""

    description = description_for(ctx.metadata_dir, scene_id)
    tags = tags_for(ctx.metadata_dir, scene_id)

    return {
        "scene_id": scene_id,
        "id": scene_id,
        "film_slug": slug,
        "keyframe_url": keyframe_url(entry.get("filepath", ""), ctx.data_dir) or "",
        "timecode": timecode,
        "start_s": start_s,
        "end_s": end_s,
        "title": None,
        "description": description,
        "tags": tags,
    }


def description_for(metadata_dir: Path, scene_id: int) -> str:
    """Look up the moondream description for ``scene_id`` (``""`` if absent)."""
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


def tags_for(metadata_dir: Path, scene_id: int) -> list[str]:
    """Return the merged (LLM + manual) tag list for ``scene_id`` (``[]`` if absent)."""
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


def resolve_timecode(cfg: Any, slug: str, scene_id: int) -> str:
    """Return the SMPTE timecode of ``(slug, scene_id)``'s start, or ``""``."""
    try:
        ctx = FilmContext.for_film(cfg, slug)
    except ValueError:
        return ""
    kf_meta = load_json(ctx.metadata_dir / "keyframes_metadata.json") or []
    if not isinstance(kf_meta, list) or not kf_meta:
        return ""
    fps = derive_fps(kf_meta)
    for entry in kf_meta:
        try:
            if int(entry.get("scene_id")) == scene_id:
                start_s = float(entry.get("start_time_s") or 0.0)
                return to_smpte(start_s, fps) if start_s > 0 else ""
        except (TypeError, ValueError):
            continue
    return ""
