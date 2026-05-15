"""Annotate tab routes — manual scene tagging."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Form, Query, Request
from fastapi.responses import HTMLResponse

from api.deps import get_config, make_ctx
from api.templates import templates

logger = logging.getLogger(__name__)
router = APIRouter()

_BROKEN_LLM = "One or two sentences about subject"


# ── Data helpers ──────────────────────────────────────────────────────────────

def _load_json(path: Path):
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def _build_scene_list(meta_dir: Path, filter_mode: str) -> tuple[list, dict, dict]:
    """Return (scene_list, desc_by_scene, annotations)."""
    from cinemateca.annotator import load as load_annotations

    kf_meta = _load_json(meta_dir / "keyframes_metadata.json") or []
    descriptions = _load_json(meta_dir / "scene_descriptions.json") or []
    annotations = load_annotations(meta_dir)

    desc_by_scene = {d["scene_id"]: d for d in descriptions if "scene_id" in d}

    valid_desc_ids = {
        d["scene_id"] for d in descriptions
        if "error" not in d and _BROKEN_LLM not in d.get("description", "")
    }

    if filter_mode == "no_llm":
        scenes = [s for s in kf_meta if s["scene_id"] not in valid_desc_ids]
    else:
        scenes = list(kf_meta)

    return scenes, desc_by_scene, annotations


def _scene_context(
    scenes: list,
    scene_id: Optional[int],
    data_dir: Path,
    desc_by_scene: dict,
    annotations: dict,
) -> dict:
    """Build template context for the annotate scene panel."""
    if not scenes:
        return {"scene": None, "scene_list": [], "total": 0, "annotated_count": 0}

    # Default to first scene if scene_id not found
    if scene_id is None or not any(s["scene_id"] == scene_id for s in scenes):
        scene_id = scenes[0]["scene_id"]

    idx = next(i for i, s in enumerate(scenes) if s["scene_id"] == scene_id)
    scene = scenes[idx]

    fp = Path(scene.get("filepath", ""))
    start_s = float(scene.get("start_time_s", 0))
    end_s = float(scene.get("end_time_s", 0))

    llm = desc_by_scene.get(scene_id)
    has_llm = bool(llm and _BROKEN_LLM not in llm.get("description", ""))

    existing_tags = annotations.get(str(scene_id), [])
    annotated_count = sum(1 for s in scenes if str(s["scene_id"]) in annotations)

    return {
        "scene": scene,
        "scene_id": scene_id,
        "img_url": _keyframe_url(fp, data_dir),
        "start_s": start_s,
        "end_s": end_s,
        "duration_s": end_s - start_s,
        "llm": llm if has_llm else None,
        "existing_tags": existing_tags,
        "tags_value": ", ".join(existing_tags),
        "prev_id": scenes[idx - 1]["scene_id"] if idx > 0 else None,
        "next_id": scenes[idx + 1]["scene_id"] if idx < len(scenes) - 1 else None,
        "current_idx": idx,
        "total": len(scenes),
        "annotated_count": annotated_count,
        "scene_list": scenes,
    }


def _keyframe_url(fp: Path, data_dir: Path) -> Optional[str]:
    for candidate in (fp, Path.cwd() / fp):
        try:
            rel = candidate.resolve().relative_to(data_dir.resolve())
            return f"/media/{rel.as_posix()}"
        except ValueError:
            continue
    return None


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/tab/annotate", response_class=HTMLResponse)
async def tab_annotate(
    request: Request,
    filter: str = Query(default="no_llm"),
    id: Optional[int] = Query(default=None),
) -> HTMLResponse:
    cfg = get_config()
    meta_dir = Path(cfg.paths.metadata_dir)
    data_dir = Path(cfg.paths.data_dir).resolve()

    no_data = not bool(_load_json(meta_dir / "keyframes_metadata.json"))
    scenes, desc_by_scene, annotations = _build_scene_list(meta_dir, filter)
    all_done = (not no_data) and (not scenes) and filter == "no_llm"

    ctx = _scene_context(scenes, id, data_dir, desc_by_scene, annotations)

    return templates.TemplateResponse(
        request,
        "partials/annotate.html",
        make_ctx(request, filter=filter, no_data=no_data, all_done=all_done, **ctx),
    )


@router.get("/api/annotate/scene", response_class=HTMLResponse)
async def api_annotate_scene(
    request: Request,
    id: int = Query(...),
    filter: str = Query(default="no_llm"),
) -> HTMLResponse:
    cfg = get_config()
    meta_dir = Path(cfg.paths.metadata_dir)
    data_dir = Path(cfg.paths.data_dir).resolve()

    scenes, desc_by_scene, annotations = _build_scene_list(meta_dir, filter)
    ctx = _scene_context(scenes, id, data_dir, desc_by_scene, annotations)

    return templates.TemplateResponse(
        request,
        "partials/annotate_scene.html",
        make_ctx(request, filter=filter, **ctx),
    )


@router.post("/api/annotate/save", response_class=HTMLResponse)
async def api_annotate_save(
    request: Request,
    scene_id: int = Form(...),
    filter: str = Form(default="no_llm"),
    tags: str = Form(default=""),
) -> HTMLResponse:
    from cinemateca.annotator import load as load_annotations, save as save_annotations

    cfg = get_config()
    meta_dir = Path(cfg.paths.metadata_dir)
    data_dir = Path(cfg.paths.data_dir).resolve()

    new_tags = [t.strip().lower().replace(" ", "-") for t in tags.split(",") if t.strip()]
    ann = load_annotations(meta_dir)
    ann[str(scene_id)] = new_tags
    save_annotations(meta_dir, ann)
    logger.info("Saved %d tag(s) for scene %s", len(new_tags), scene_id)

    scenes, desc_by_scene, annotations = _build_scene_list(meta_dir, filter)
    ctx = _scene_context(scenes, scene_id, data_dir, desc_by_scene, annotations)

    return templates.TemplateResponse(
        request,
        "partials/annotate_scene.html",
        make_ctx(request, filter=filter, saved=True, **ctx),
    )


@router.post("/api/annotate/clear", response_class=HTMLResponse)
async def api_annotate_clear(
    request: Request,
    scene_id: int = Form(...),
    filter: str = Form(default="no_llm"),
) -> HTMLResponse:
    from cinemateca.annotator import load as load_annotations, save as save_annotations

    cfg = get_config()
    meta_dir = Path(cfg.paths.metadata_dir)
    data_dir = Path(cfg.paths.data_dir).resolve()

    ann = load_annotations(meta_dir)
    ann.pop(str(scene_id), None)
    save_annotations(meta_dir, ann)
    logger.info("Cleared tags for scene %s", scene_id)

    scenes, desc_by_scene, annotations = _build_scene_list(meta_dir, filter)
    ctx = _scene_context(scenes, scene_id, data_dir, desc_by_scene, annotations)

    return templates.TemplateResponse(
        request,
        "partials/annotate_scene.html",
        make_ctx(request, filter=filter, cleared=True, **ctx),
    )
