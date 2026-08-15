"""Read/write the ``scene_motion.json`` artefact.

Purely additive: this file sits alongside the existing per-film metadata
and nothing already on disk changes shape. A library without it behaves
exactly as before — every reader here degrades to empty rather than
raising, so motion is an enhancement and never a dependency.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from kuaa.motion.analyzer import SceneMotion, motion_tag_index

logger = logging.getLogger(__name__)

MOTION_FILENAME = "scene_motion.json"


def save_motion(metadata_dir: Path, records: list[SceneMotion]) -> Path:
    """Write ``scene_motion.json`` atomically. Returns the written path."""
    path = Path(metadata_dir) / MOTION_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "scenes": [record.to_dict() for record in records],
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    os.replace(tmp, path)
    return path


def load_motion(metadata_dir: Path) -> list[SceneMotion]:
    """Read ``scene_motion.json``; ``[]`` when absent or malformed."""
    path = Path(metadata_dir) / MOTION_FILENAME
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        logger.warning("motion: malformed %s — treating as absent", path)
        return []
    scenes = raw.get("scenes") if isinstance(raw, dict) else None
    if not isinstance(scenes, list):
        return []
    out: list[SceneMotion] = []
    for row in scenes:
        if not isinstance(row, dict):
            continue
        try:
            out.append(
                SceneMotion(
                    scene_id=int(row["scene_id"]),
                    shot_length_s=float(row.get("shot_length_s", 0.0)),
                    motion_magnitude=float(row.get("motion_magnitude", 0.0)),
                    camera_motion=str(row.get("camera_motion", "static")),
                    subject_motion=str(row.get("subject_motion", "still")),
                    camera_share=float(row.get("camera_share", 0.0)),
                    sampled_pairs=int(row.get("sampled_pairs", 0)),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return out


def load_motion_tag_index(metadata_dir: Path) -> dict[str, list[int]]:
    """Motion tags in ``{tag: [scene_id, ...]}`` form; ``{}`` when absent.

    Same shape as ``scene_tags.json``, so callers merge it through the
    existing tag machinery instead of a parallel path.
    """
    records = load_motion(metadata_dir)
    return motion_tag_index(records) if records else {}
