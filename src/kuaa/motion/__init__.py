"""Per-scene motion signal — the one thing every other model is blind to.

SigLIP, YOLOv8, MTCNN and Moondream all consume a single still JPEG, so
nothing downstream of ``scene_cuts.json`` can tell a static tableau from a
whip pan. This package adds that axis from the source video, on CPU,
without re-running any model.

Public surface:

``SceneMotion``      — per-scene statistics record.
``analyze_video``    — compute records for one film's scene boundaries.
``motion_tag_index`` — invert records into ``{tag: [scene_id, ...]}``.
``load_motion`` / ``save_motion`` — the ``scene_motion.json`` artefact.
"""

from __future__ import annotations

from kuaa.motion.analyzer import (
    DEFAULT_SAMPLE_FPS,
    SceneMotion,
    analyze_video,
    motion_tag_index,
)
from kuaa.motion.io import MOTION_FILENAME, load_motion, load_motion_tag_index, save_motion

__all__ = [
    "DEFAULT_SAMPLE_FPS",
    "MOTION_FILENAME",
    "SceneMotion",
    "analyze_video",
    "load_motion",
    "load_motion_tag_index",
    "motion_tag_index",
    "save_motion",
]
