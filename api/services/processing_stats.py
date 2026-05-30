"""Processing tab — stats aggregation and resource metrics.

Extracted from ``api/services/processing_service.py`` (A2/G1 split).
``processing_service.py`` re-imports these symbols so existing
``from api.services.processing_service import ...`` call sites are unchanged.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import MappingProxyType
from typing import Any

logger = logging.getLogger(__name__)


# ── Per-step substep recipes ──────────────────────────────────────────────────
#
# Kept here alongside ``_fallback_substeps`` because ``build_active_step``
# (which remains in processing_service) calls ``_fallback_substeps``, and
# we re-export it from processing_service to keep the public surface intact.

_SUBSTEP_RECIPES: MappingProxyType[str, tuple[dict[str, str], ...]] = MappingProxyType(
    {
        "frame_extraction": (
            {"label": "Probe video", "sub": "ffmpeg · streams", "value": "~0.3s"},
            {"label": "Extract frames", "sub": "1 fps · 480p", "value": ""},
            {"label": "Persist frames", "sub": "data/frames/", "value": "~5s"},
        ),
        "scene_detection": (
            {"label": "Read frames", "sub": "PySceneDetect input", "value": "~0.5s"},
            {"label": "Detect cuts", "sub": "adaptive threshold", "value": ""},
            {"label": "Pick keyframes", "sub": "1 per scene", "value": "~2s"},
            {"label": "Persist scenes", "sub": "keyframes_metadata.json", "value": "~1s"},
        ),
        "visual_analysis": (
            {"label": "Load YOLOv8 weights", "sub": "ultralytics", "value": "~0.4s"},
            {"label": "Load MTCNN", "sub": "facenet-pytorch", "value": "~0.6s"},
            {"label": "Detect objects", "sub": "per scene", "value": ""},
            {"label": "Detect faces", "sub": "per scene", "value": ""},
            {"label": "Persist tags", "sub": "tags_per_scene.json", "value": "~15s"},
        ),
        "embeddings": (
            {"label": "Load CLIP", "sub": "ViT-B/32", "value": "~1s"},
            {"label": "Encode keyframes", "sub": "batch · GPU/CPU", "value": ""},
            {"label": "Persist embeddings", "sub": "keyframe_embeddings.npy", "value": "~2s"},
        ),
        "llm_description": (
            {"label": "Load Moondream2", "sub": "transformers", "value": "~6s"},
            {"label": "Describe scenes", "sub": "per keyframe", "value": ""},
            {"label": "Persist descriptions", "sub": "scene_descriptions.json", "value": "~3s"},
        ),
    }
)

_DEFAULT_RECIPE: tuple[dict[str, str], ...] = (
    {"label": "Load weights", "sub": "", "value": ""},
    {"label": "Run", "sub": "", "value": ""},
    {"label": "Persist outputs", "sub": "", "value": ""},
)

_ACTIVE_STATUS_BY_INDEX = ("done", "active")  # i=0 → done, i=1 → active, else pending


def _fallback_substeps(step_name: str, step_state: str) -> list[dict[str, str]]:
    """Per-step synthetic substeps so the right pane renders meaningfully.

    Status derived from the parent step's state:
      'done'   → all done
      'active' → first sub done, second active, remainder pending
      else     → all pending
    """
    recipe = _SUBSTEP_RECIPES.get(step_name, _DEFAULT_RECIPE)
    if step_state == "done":
        return [{**r, "status": "done"} for r in recipe]
    if step_state == "active":
        return [
            {**r, "status": _ACTIVE_STATUS_BY_INDEX[i] if i < 2 else "pending"}
            for i, r in enumerate(recipe)
        ]
    return [{**r, "status": "pending"} for r in recipe]


# ── Stats aggregation ─────────────────────────────────────────────────────────


def aggregate_stats(library_dir: Path) -> dict[str, Any]:
    """Sum scene-level counts across all registered films.

    Reads each film's ``metadata/keyframes_metadata.json`` (scene count)
    and best-effort sums embeddings/descriptions/faces/objects when the
    metadata files are present. Anything missing stays at 0.

    Frames are not persisted per-scene in metadata; ``frames`` falls
    back to ``scenes * 1`` (rough lower bound) so the card never shows
    a less-than-scenes number. A future pass can hook this into the
    real frame index for an honest count.
    """
    import json

    stats = {
        "frames": 0,
        "scenes": 0,
        "embeddings": 0,
        "descriptions": 0,
        "faces": 0,
        "objects": 0,
        "faces_warn": False,
    }

    if not library_dir.exists():
        return stats

    for film_dir in sorted(library_dir.iterdir()):
        if not film_dir.is_dir():
            continue

        meta_dir = film_dir / "metadata"
        kf_path = meta_dir / "keyframes_metadata.json"
        film_scene_count = 0
        if kf_path.exists():
            try:
                with open(kf_path, encoding="utf-8") as f:
                    kf_meta = json.load(f)
                if isinstance(kf_meta, list):
                    film_scene_count = len(kf_meta)
                    stats["scenes"] += film_scene_count
                    # Lower-bound frame estimate (one per scene) — honest
                    # placeholder until a real frame count lands.
                    stats["frames"] += film_scene_count
            except (json.JSONDecodeError, OSError):
                pass

        emb_path = film_dir / "embeddings" / "keyframe_embeddings.npy"
        if emb_path.exists():
            # File presence is a proxy: the actual row count would need
            # numpy; treat one present index as N scenes worth of
            # embeddings (matches the scene count for that film, which
            # we already computed above so no second JSON read is needed).
            stats["embeddings"] += film_scene_count

        desc_path = meta_dir / "scene_descriptions.json"
        if desc_path.exists():
            try:
                with open(desc_path, encoding="utf-8") as f:
                    descs = json.load(f)
                if isinstance(descs, (list, dict)):
                    stats["descriptions"] += len(descs)
            except (json.JSONDecodeError, OSError):
                pass

        tags_path = meta_dir / "scene_tags.json"
        if tags_path.exists():
            try:
                with open(tags_path, encoding="utf-8") as f:
                    tags = json.load(f)
                # scene_tags.json is typically a {scene_id: [tags]} dict
                # or a list of per-scene records — count faces/objects
                # by tag-name match. Defensive: any structural surprise
                # is silently skipped.
                if isinstance(tags, dict):
                    for scene_tags in tags.values():
                        if isinstance(scene_tags, list):
                            for t in scene_tags:
                                name = (t or "").lower() if isinstance(t, str) else ""
                                if "face" in name:
                                    stats["faces"] += 1
                                elif name and name not in {"day", "night", "indoor", "outdoor"}:
                                    stats["objects"] += 1
            except (json.JSONDecodeError, OSError):
                pass

    return stats


# ── Resource metrics ────────────────────────────────────────────────────────────


def _metric(label: str, value: float) -> dict[str, Any]:
    """Clamp a 0..1 metric value into the shape the resource card renders."""
    return {"label": label, "value": max(0.0, min(1.0, float(value)))}


def build_resource_metrics() -> list[dict[str, Any]]:
    """Return local CPU/RAM and optional accelerator memory metrics.

    The function is deliberately best-effort. It never imports torch solely for
    the UI card, because torch import can dominate a page refresh; if the
    pipeline has already loaded torch and CUDA is available, VRAM appears too.
    """
    metrics: list[dict[str, Any]] = []
    try:
        import psutil

        metrics.append(_metric("CPU", psutil.cpu_percent(interval=0.0) / 100.0))
        metrics.append(_metric("RAM", psutil.virtual_memory().percent / 100.0))
    except (ImportError, OSError):
        pass

    try:
        import sys

        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            if total:
                metrics.append(_metric("VRAM", 1.0 - (float(free) / float(total))))
    except Exception as exc:  # pragma: no cover - hardware/runtime dependent
        logger.debug("resource metric probe skipped: %s", exc)

    return metrics
