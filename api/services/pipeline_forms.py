"""Shared form handling for the pipeline start/enqueue routes.

``/api/pipeline/start`` and ``/api/pipeline/enqueue`` accept an identical
multipart form (video path + step selection + scene-detection overrides)
and differ only in what they do with the resolved values. Extracted here
to keep ``api/routes/processing.py`` within its LOC budget.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import Form, Request

from api.deps import request_gettext
from api.jobs import STEP_DEFS
from api.services.processing_render import processing_tab_response


class PipelineFormParams:
    """FastAPI dependency bundling the pipeline start/enqueue form fields."""

    def __init__(
        self,
        video_path: str = Form(...),
        steps: list[str] = Form(default=[]),
        sd_detector: Literal["content", "adaptive"] = Form(default="adaptive"),
        sd_adaptive_threshold: float = Form(default=3.0),
        sd_content_threshold: float = Form(default=27.0),
        sd_min_scene_len: int = Form(default=15),
        sd_keyframes_per_scene: int = Form(default=3),
        sd_keyframe_height: int = Form(default=480),
    ) -> None:
        self.video_path = video_path
        self.steps = steps or [name for name, _ in STEP_DEFS]
        self.sd_detector = sd_detector
        self.sd_adaptive_threshold = sd_adaptive_threshold
        self.sd_content_threshold = sd_content_threshold
        self.sd_min_scene_len = sd_min_scene_len
        self.sd_keyframes_per_scene = sd_keyframes_per_scene
        self.sd_keyframe_height = sd_keyframe_height


def sd_override_from_fields(
    cfg,
    *,
    detector: Literal["content", "adaptive"],
    adaptive_threshold: float,
    content_threshold: float,
    min_scene_len: int,
    keyframes_per_scene: int,
    keyframe_height: int,
):
    """Return a SceneDetectionCfg override only when values differ from cfg.

    Value-based so both the Processing form (:func:`build_sd_override`) and the
    Pre-processing detect route share one construction + equality check.
    """
    from kuaa.config.schema import SceneDetectionCfg

    override = SceneDetectionCfg(
        detector=detector,
        adaptive_threshold=adaptive_threshold,
        content_threshold=content_threshold,
        min_scene_len=min_scene_len,
        keyframes_per_scene=keyframes_per_scene,
        keyframe_height=keyframe_height,
    )
    return None if override == cfg.scene_detection else override


def build_sd_override(cfg, params: PipelineFormParams):
    """Return a SceneDetectionCfg override only when values differ from cfg."""
    return sd_override_from_fields(
        cfg,
        detector=params.sd_detector,
        adaptive_threshold=params.sd_adaptive_threshold,
        content_threshold=params.sd_content_threshold,
        min_scene_len=params.sd_min_scene_len,
        keyframes_per_scene=params.sd_keyframes_per_scene,
        keyframe_height=params.sd_keyframe_height,
    )


def resolve_pipeline_request(request: Request, cfg, params: PipelineFormParams, route_name: str):
    """Validate ``params.video_path`` and build the scene-detection override.

    Returns ``(video_path, steps, sd_override)`` on success, or an
    ``HTMLResponse`` (rendered error toast) when the path does not exist.
    """
    import logging

    logger = logging.getLogger(__name__)
    vp = Path(params.video_path)
    _ = request_gettext(request)
    if not vp.exists():
        logger.warning("%s rejected — file not found: %s", route_name, vp)
        file_not_found = _("File not found. Check the path or filename.")
        return processing_tab_response(request, error_message=file_not_found)
    return vp, set(params.steps), build_sd_override(cfg, params)
