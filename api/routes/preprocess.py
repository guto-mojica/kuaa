"""Pre-processing tab routes — scene-detection runs + cut review/editing.

Scene detection is detached from the Processing tab into its own workspace so
operators can get scene boundaries right before the expensive downstream steps.
Detection runs reuse the Processing job registry + SSE stream
(``/api/pipeline/stream``); this module adds only the SD-only launch and the
manual split/merge edit endpoints.
"""

from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, Depends, Form, Request, Response
from fastapi.responses import FileResponse, HTMLResponse

from api.deps import film_slug_query, get_config, make_ctx, resolve_film_context
from api.jobs import ConcurrencyRejected, start_job
from api.services.pipeline_forms import sd_override_from_fields
from api.services.preprocess_render import (
    PREPROCESS_STEPS,
    build_preprocess_context,
    build_preprocess_start_response,
    film_video_path,
    render_filmstrip_fragment,
)
from api.templates import templates
from kuaa.preprocess import (
    CutEditError,
    apply_pending,
    discard_pending,
    stage_merge,
    stage_split,
    toggle_pending,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/tab/pre-processing", response_class=HTMLResponse)
async def tab_preprocessing(
    request: Request, slug: str | None = Depends(film_slug_query)
) -> HTMLResponse:
    ctx = build_preprocess_context(slug)
    logger.info("/tab/pre-processing — slug=%s", slug)
    return templates.TemplateResponse(
        request, "partials/preprocess.html", make_ctx(request, current_slug=slug, **ctx)
    )


@router.post("/api/preprocess/detect", response_class=HTMLResponse)
async def api_preprocess_detect(
    request: Request,
    film: str = Form(...),
    sd_detector: Literal["content", "adaptive"] = Form(default="adaptive"),
    sd_adaptive_threshold: float = Form(default=3.0),
    sd_content_threshold: float = Form(default=27.0),
    sd_min_scene_len: int = Form(default=15),
    sd_keyframes_per_scene: int = Form(default=3),
    sd_keyframe_height: int = Form(default=480),
) -> HTMLResponse:
    """Start a scene-detection-only run for the selected film."""
    cfg = get_config()
    vp = film_video_path(cfg, film)
    if vp is None:
        logger.warning("/api/preprocess/detect rejected — no video for slug: %s", film)
        return HTMLResponse('<p class="text-error">File not found.</p>', status_code=400)
    sd_override = sd_override_from_fields(
        cfg,
        detector=sd_detector,
        adaptive_threshold=sd_adaptive_threshold,
        content_threshold=sd_content_threshold,
        min_scene_len=sd_min_scene_len,
        keyframes_per_scene=sd_keyframes_per_scene,
        keyframe_height=sd_keyframe_height,
    )
    try:
        job_id = start_job(str(vp), set(PREPROCESS_STEPS), cfg, sd_override)
    except ConcurrencyRejected as exc:
        return HTMLResponse(f'<p class="text-error">{exc}</p>', status_code=409)
    logger.info("/api/preprocess/detect — accepted job_id=%s", job_id)
    return build_preprocess_start_response(request, cfg, vp, film)


_VIDEO_MEDIA_TYPES = {
    ".mp4": "video/mp4",
    ".m4v": "video/x-m4v",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
}


@router.get("/api/preprocess/video/{slug}")
async def api_preprocess_video(slug: str) -> Response:
    """Stream a film's source video for the review player (range-seekable).

    ``FileResponse`` honours the ``Range`` header and reads the resolved file
    directly, so a source symlinked outside ``data_dir`` still plays.
    """
    cfg = get_config()
    video_path = film_video_path(cfg, slug)
    if video_path is None:
        return HTMLResponse('<p class="text-error">Source video not found.</p>', status_code=404)
    media_type = _VIDEO_MEDIA_TYPES.get(video_path.suffix.lower(), "application/octet-stream")
    return FileResponse(video_path, media_type=media_type)


@router.get("/api/preprocess/filmstrip", response_class=HTMLResponse)
async def api_preprocess_filmstrip(
    request: Request, slug: str | None = Depends(film_slug_query)
) -> HTMLResponse:
    """Return the filmstrip fragment for ``slug`` (refresh after a run/edit)."""
    return render_filmstrip_fragment(request, slug or "")


@router.post("/api/preprocess/cut/split", response_class=HTMLResponse)
async def api_preprocess_split(
    request: Request, slug: str = Form(...), at_frame: int = Form(...)
) -> HTMLResponse:
    """Stage a split at ``at_frame`` — cheap JSON write, no rebuild until Apply."""
    return _run_cut_route(request, slug, lambda ctx, cfg, vp: stage_split(ctx, at_frame=at_frame))


@router.post("/api/preprocess/cut/merge", response_class=HTMLResponse)
async def api_preprocess_merge(
    request: Request, slug: str = Form(...), cut_frame: int = Form(...)
) -> HTMLResponse:
    """Stage a merge removing the cut at ``cut_frame`` — cheap, no rebuild until Apply."""
    return _run_cut_route(request, slug, lambda ctx, cfg, vp: stage_merge(ctx, cut_frame=cut_frame))


@router.post("/api/preprocess/cut/pending/{index}/toggle", response_class=HTMLResponse)
async def api_preprocess_pending_toggle(
    request: Request, index: int, slug: str = Form(...)
) -> HTMLResponse:
    """Flip whether a single staged edit will be applied."""
    return _run_cut_route(request, slug, lambda ctx, cfg, vp: toggle_pending(ctx, index=index))


@router.post("/api/preprocess/cut/discard", response_class=HTMLResponse)
async def api_preprocess_discard(request: Request, slug: str = Form(...)) -> HTMLResponse:
    """Clear the staged-edit session — no rebuild, no-op if nothing staged."""
    return _run_cut_route(request, slug, lambda ctx, cfg, vp: discard_pending(ctx))


@router.post("/api/preprocess/cut/apply", response_class=HTMLResponse)
async def api_preprocess_apply(request: Request, slug: str = Form(...)) -> HTMLResponse:
    """Apply the checked staged edits — the real (now partial) rebuild."""
    return _run_cut_route(
        request, slug, lambda ctx, cfg, vp: apply_pending(ctx, cfg=cfg, video_path=vp)
    )


def _run_cut_route(request: Request, slug: str, op) -> HTMLResponse:
    """Resolve ``slug``'s ``FilmContext`` + confirm its video is on disk (every
    cut route shares this guard, even staging ones that don't touch the video
    themselves — editing is only ever exposed in the UI when the source is
    present), run ``op(ctx, cfg, video_path)``, and render the refreshed
    filmstrip fragment. ``op`` raises ``CutEditError`` for an invalid edit,
    mapped to a 422.
    """
    cfg = get_config()
    video_path = film_video_path(cfg, slug)
    if video_path is None:
        return HTMLResponse('<p class="text-error">Film not found.</p>', status_code=404)
    try:
        ctx = resolve_film_context(cfg, slug, None)
    except ValueError:
        return HTMLResponse('<p class="text-error">Film not registered.</p>', status_code=404)
    try:
        filmstrip = op(ctx, cfg, video_path)
    except CutEditError as exc:
        logger.warning("/api/preprocess/cut — %s", exc)
        return HTMLResponse(f'<p class="text-error">{exc}</p>', status_code=422)
    return templates.TemplateResponse(
        request,
        "partials/preprocess_filmstrip.html",
        make_ctx(request, filmstrip=filmstrip, pp_slug=slug, pp_title=""),
    )
