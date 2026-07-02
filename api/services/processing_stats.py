"""Processing tab — stats aggregation and resource metrics.

Extracted from ``api/services/processing_service.py`` (A2/G1 split).
``processing_service.py`` re-imports these symbols so existing
``from api.services.processing_service import ...`` call sites are unchanged.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Pipeline step descriptions (right-pane copy) ──────────────────────────────
#
# Short, factual descriptions of what each pipeline step does — surfaced in the
# .p-rp "what" paragraph and as the label/detail of each real substep row.

_STEP_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "frame_extraction": {
        "label": "Frame extraction",
        "detail": "ffmpeg · 1 fps · 480p",
        "description": (
            "Decode the source video and emit one keyframe per second "
            "(downscaled). Feeds scene detection and visual analysis."
        ),
    },
    "scene_detection": {
        "label": "Scene detection",
        "detail": "PySceneDetect · adaptive",
        "description": (
            "Detect shot boundaries across the extracted frames and "
            "build the scene index. Output: scene cuts + representative "
            "keyframes."
        ),
    },
    "visual_analysis": {
        "label": "Visual analysis",
        "detail": "YOLOv8 (objects) + MTCNN (faces)",
        "description": (
            "Run object and face detection on each scene's keyframes to "
            "produce automatic tags. Output feeds embeddings + descriptions."
        ),
    },
    "embeddings": {
        "label": "Embeddings",
        "detail": "CLIP ViT-B/32",
        "description": (
            "Encode every keyframe into a CLIP embedding so the scene "
            "is searchable by text query or by another image."
        ),
    },
    "llm_description": {
        "label": "LLM descriptions",
        "detail": "Moondream2 · transformers",
        "description": (
            "Generate a short natural-language description of each scene "
            "from its keyframe. Slowest step; CPU-bound by default."
        ),
    },
}


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
