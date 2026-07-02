"""Processing tab routes — pipeline start, SSE stream, status."""

from __future__ import annotations

import logging
import uuid as _uuid
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from api.deps import film_slug_query, get_config, make_ctx, request_gettext
from api.jobs import (
    STEP_DEFS,
    ConcurrencyRejected,
    cancel_job,
    get_job,
    queue_job,
    remove_pending_job,
    start_job,
    start_queued_jobs,
)
from api.services.processing_render import (
    build_processing_context,
    build_sse_generator,
    build_start_response,
    processing_tab_response,
)
from api.services.processing_service import enrich_jobs
from api.templates import templates

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/tab/processing", response_class=HTMLResponse)
async def tab_processing(
    request: Request, slug: str | None = Depends(film_slug_query)
) -> HTMLResponse:
    ctx = build_processing_context()
    logger.info("/tab/processing — slug=%s jobs=%d", slug, len(ctx["jobs"]))
    return templates.TemplateResponse(
        request, "partials/processing.html", make_ctx(request, current_slug=slug, **ctx)
    )


def _build_sd_override(
    cfg,
    sd_detector: Literal["content", "adaptive"],
    sd_adaptive_threshold: float,
    sd_content_threshold: float,
    sd_min_scene_len: int,
    sd_keyframes_per_scene: int,
    sd_keyframe_height: int,
):
    """Return a SceneDetectionCfg override only when values differ from cfg."""
    from kuaa.config.schema import SceneDetectionCfg

    current = cfg.scene_detection
    override = SceneDetectionCfg(
        detector=sd_detector,
        adaptive_threshold=sd_adaptive_threshold,
        content_threshold=sd_content_threshold,
        min_scene_len=sd_min_scene_len,
        keyframes_per_scene=sd_keyframes_per_scene,
        keyframe_height=sd_keyframe_height,
    )
    if override == current:
        return None
    return override


@router.post("/api/pipeline/start", response_class=HTMLResponse)
async def api_pipeline_start(
    request: Request,
    video_path: str = Form(...),
    steps: list[str] = Form(default=[]),
    sd_detector: Literal["content", "adaptive"] = Form(default="adaptive"),
    sd_adaptive_threshold: float = Form(default=3.0),
    sd_content_threshold: float = Form(default=27.0),
    sd_min_scene_len: int = Form(default=15),
    sd_keyframes_per_scene: int = Form(default=3),
    sd_keyframe_height: int = Form(default=480),
) -> HTMLResponse:
    if not steps:
        steps = [name for name, _ in STEP_DEFS]
    cfg = get_config()
    vp = Path(video_path)
    _ = request_gettext(request)
    file_not_found = _("File not found. Check the path or filename.")
    if not vp.exists():
        logger.warning("/api/pipeline/start rejected — file not found: %s", vp)
        return processing_tab_response(request, error_message=file_not_found)
    sd_override = _build_sd_override(
        cfg,
        sd_detector,
        sd_adaptive_threshold,
        sd_content_threshold,
        sd_min_scene_len,
        sd_keyframes_per_scene,
        sd_keyframe_height,
    )
    try:
        job_id = start_job(str(vp), set(steps), cfg, sd_override)
    except ConcurrencyRejected as exc:
        return HTMLResponse(f'<p class="text-error">{exc}</p>', status_code=409)
    logger.info("/api/pipeline/start — accepted job_id=%s", job_id)
    return build_start_response(request, cfg, vp, request.cookies.get("active_film", ""))


@router.post(
    "/api/pipeline/enqueue", response_class=HTMLResponse
)  # queue only, never auto-starts; use /queue/start
async def api_pipeline_enqueue(
    request: Request,
    video_path: str = Form(...),
    steps: list[str] = Form(default=[]),
    sd_detector: Literal["content", "adaptive"] = Form(default="adaptive"),
    sd_adaptive_threshold: float = Form(default=3.0),
    sd_content_threshold: float = Form(default=27.0),
    sd_min_scene_len: int = Form(default=15),
    sd_keyframes_per_scene: int = Form(default=3),
    sd_keyframe_height: int = Form(default=480),
) -> HTMLResponse:
    if not steps:
        steps = [name for name, _ in STEP_DEFS]
    cfg = get_config()
    vp = Path(video_path)
    _ = request_gettext(request)
    file_not_found = _("File not found. Check the path or filename.")
    if not vp.exists():
        logger.warning("/api/pipeline/enqueue rejected — file not found: %s", vp)
        return processing_tab_response(request, error_message=file_not_found)
    sd_override = _build_sd_override(
        cfg,
        sd_detector,
        sd_adaptive_threshold,
        sd_content_threshold,
        sd_min_scene_len,
        sd_keyframes_per_scene,
        sd_keyframe_height,
    )
    queue_job(str(vp), set(steps), cfg, sd_override)
    logger.info("/api/pipeline/enqueue — queued %s", vp)
    return build_start_response(request, cfg, vp, request.cookies.get("active_film", ""))


@router.post("/api/pipeline/queue/start", response_class=HTMLResponse)  # no-op when busy/empty
async def api_pipeline_queue_start(request: Request) -> HTMLResponse:
    job_id = start_queued_jobs()
    logger.info("/api/pipeline/queue/start — job_id=%s", job_id)
    ctx = build_processing_context()
    return templates.TemplateResponse(request, "partials/processing.html", make_ctx(request, **ctx))


@router.post("/api/pipeline/pending/{entry_id}/remove", response_class=HTMLResponse)
async def api_pipeline_pending_remove(
    request: Request, entry_id: str
) -> HTMLResponse:  # returns refreshed queue fragment
    removed = remove_pending_job(entry_id)
    if not removed:
        logger.warning("/api/pipeline/pending/remove — entry %s not found", entry_id)
    ctx = build_processing_context()
    return templates.TemplateResponse(
        request, "partials/processing_queue.html", make_ctx(request, **ctx)
    )


@router.post("/api/pipeline/cancel/{job_id}", response_class=HTMLResponse)
async def api_pipeline_cancel(
    request: Request, job_id: str
) -> HTMLResponse:  # cooperative cancellation
    ok = cancel_job(job_id)
    job = get_job(job_id)
    if job is None:
        return HTMLResponse('<p class="text-error">Job not found.</p>', status_code=404)
    if not ok:
        return HTMLResponse('<p class="text-muted">Job already finished.</p>', status_code=409)
    enrich_jobs([job])
    return templates.TemplateResponse(
        request, "partials/processing_job.html", make_ctx(request, job=job)
    )


@router.get(
    "/api/pipeline/job-card/{job_id}", response_class=HTMLResponse
)  # full .p-active card for polling refresh
async def api_pipeline_job_card(request: Request, job_id: str) -> HTMLResponse:
    job = get_job(job_id)
    if job is None:
        return HTMLResponse('<p class="text-error">Job not found.</p>', status_code=404)
    enrich_jobs([job])
    return templates.TemplateResponse(
        request, "partials/processing_job.html", make_ctx(request, job=job)
    )


@router.get(
    "/api/pipeline/stream/{job_id}"
)  # SSE: log / update / done|error|cancelled (terminal closes stream)
async def api_pipeline_stream(request: Request, job_id: str) -> StreamingResponse:
    return StreamingResponse(
        build_sse_generator(job_id, request.cookies.get("locale", "pt_BR")),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Request-ID": getattr(request.state, "request_id", None) or str(_uuid.uuid4()),
        },
    )
