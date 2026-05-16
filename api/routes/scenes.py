"""Scenes tab routes — catalogue browsing with tag and keyword filters."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse

from api.deps import get_config, make_ctx
from api.templates import templates
from cinemateca.scene_ids import scene_id_key

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Data helpers ──────────────────────────────────────────────────────────────

def _load_json(path: Path) -> list | dict | None:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def _load_metadata(metadata_dir: Path) -> tuple[list, dict, dict, dict]:
    """Return (kf_meta, desc_by_scene, vis_by_scene, tag_index)."""
    from cinemateca.annotator import load as load_annotations, merge_tag_index
    from cinemateca.scene_ids import normalize_tag_index

    kf_meta = _load_json(metadata_dir / "keyframes_metadata.json") or []
    descriptions = _load_json(metadata_dir / "scene_descriptions.json") or []
    llm_tags = _load_json(metadata_dir / "scene_tags.json") or {}
    visual_data = _load_json(metadata_dir / "visual_analysis.json") or []
    annotations = load_annotations(metadata_dir)

    desc_by_scene = {scene_id_key(d["scene_id"]): d for d in descriptions if "scene_id" in d}
    vis_by_scene = {scene_id_key(v["scene_id"]): v for v in visual_data if "scene_id" in v}
    # merge_tag_index yields a hybrid index with mixed int (LLM) / str
    # (manual) value types. Normalize to canonical str ids here so every
    # downstream membership test is str-vs-str.
    tag_index = normalize_tag_index(merge_tag_index(llm_tags, annotations))

    return kf_meta, desc_by_scene, vis_by_scene, tag_index


def _build_cards(
    kf_meta: list,
    desc_by_scene: dict,
    vis_by_scene: dict,
    tag_index: dict,
    data_dir: Path,
    selected_tags: list[str],
    keyword: str,
) -> list[dict]:
    """Filter kf_meta and build scene card dicts for the template."""
    scenes = list(kf_meta)

    # Tag filter — intersect scene_ids across all selected tags.
    # tag_index is already normalized to {tag: {canonical str id}} by
    # _load_metadata, so the membership test is str-vs-str.
    if selected_tags and tag_index:
        valid_ids = set(tag_index.get(selected_tags[0], set()))
        for tag in selected_tags[1:]:
            valid_ids &= set(tag_index.get(tag, set()))
        scenes = [s for s in scenes if scene_id_key(s.get("scene_id", "")) in valid_ids]

    # Keyword filter — search description text blob
    if keyword:
        kw = keyword.lower()
        filtered = []
        for s in scenes:
            sid = scene_id_key(s.get("scene_id", ""))
            desc = desc_by_scene.get(sid, {})
            blob = " ".join(str(v) for v in desc.values()).lower()
            if kw in blob:
                filtered.append(s)
        scenes = filtered

    cards = []
    for s in scenes:
        sid = scene_id_key(s.get("scene_id", ""))
        fp = Path(s.get("filepath", ""))
        img_url = _keyframe_url(fp, data_dir)
        tc = s.get("timecode_start") or s.get("start_timecode", "")

        # Tags from tag_index (inverted lookup). tag_index ids are already
        # canonical str keys, so this is a direct str-vs-str membership.
        scene_tags = sorted({
            tag for tag, ids in tag_index.items()
            if sid in ids
        })

        # Visual analysis summary
        vis = vis_by_scene.get(sid, {})
        env = vis.get("environment", {})
        env_parts = [p for p in [env.get("location", ""), env.get("time_of_day", "")] if p]
        num_people = vis.get("num_faces")

        # Description one-liner
        desc = desc_by_scene.get(sid, {})
        description = desc.get("description") or ""

        cards.append({
            "scene_id": s.get("scene_id"),
            "img_url": img_url,
            "timecode": tc,
            "tags": scene_tags[:8],
            "environment": " · ".join(env_parts),
            "num_people": num_people,
            "description": description[:120] if description else "",
        })

    return cards


def _keyframe_url(fp: Path, data_dir: Path) -> Optional[str]:
    for candidate in (fp, Path.cwd() / fp):
        try:
            rel = candidate.resolve().relative_to(data_dir.resolve())
            return f"/media/{rel.as_posix()}"
        except ValueError:
            continue
    return None


# ── Routes ────────────────────────────────────────────────────────────────────

def build_scenes_context() -> dict:
    """Build the template context the scenes tab partial needs.

    Shared by the ``/tab/scenes`` HTMX fragment and the ``/scenes``
    full-page route so both render identical markup (including the
    empty-state hint when no keyframes exist).
    """
    cfg = get_config()
    meta_dir = Path(cfg.paths.metadata_dir)
    data_dir = Path(cfg.paths.data_dir).resolve()

    kf_meta, desc_by_scene, vis_by_scene, tag_index = _load_metadata(meta_dir)
    available_tags = sorted(tag_index.keys())
    cards = _build_cards(kf_meta, desc_by_scene, vis_by_scene, tag_index, data_dir, [], "")

    return {"cards": cards, "available_tags": available_tags, "no_data": not kf_meta}


@router.get("/tab/scenes", response_class=HTMLResponse)
async def tab_scenes(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "partials/scenes.html",
        make_ctx(request, **build_scenes_context()),
    )


@router.get("/api/scenes", response_class=HTMLResponse)
async def api_scenes(
    request: Request,
    tags: list[str] = Query(default=[]),
    q: str = "",
) -> HTMLResponse:
    cfg = get_config()
    meta_dir = Path(cfg.paths.metadata_dir)
    data_dir = Path(cfg.paths.data_dir).resolve()

    kf_meta, desc_by_scene, vis_by_scene, tag_index = _load_metadata(meta_dir)
    cards = _build_cards(kf_meta, desc_by_scene, vis_by_scene, tag_index, data_dir, tags, q)

    return templates.TemplateResponse(
        request,
        "partials/scenes_grid.html",
        make_ctx(request, cards=cards),
    )
