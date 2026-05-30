"""Processing tab routes — pipeline start, SSE stream, status.

``/tab/processing`` accepts ``?film=<slug>`` (T9; shows the global queue).
"""

from __future__ import annotations

import asyncio
import logging
import queue
from pathlib import Path

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from api.deps import film_slug_query, get_config, make_ctx
from api.jobs import (
    STEP_DEFS,
    ConcurrencyRejected,
    cancel_job,
    get_job,
    start_job,
)
from api.services.processing_render import (
    build_processing_context,
    build_start_response,
    render_log_row,
    render_stepper,
)
from api.templates import templates

_TERMINAL_EVENTS = ("done", "error", "cancelled")

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/tab/processing", response_class=HTMLResponse)
async def tab_processing(
    request: Request,
    slug: str | None = Depends(film_slug_query),
) -> HTMLResponse:
    ctx = build_processing_context()
    logger.info("/tab/processing — slug=%s jobs=%d", slug, len(ctx["jobs"]))
    return templates.TemplateResponse(
        request, "partials/processing.html", make_ctx(request, current_slug=slug, **ctx)
    )


@router.post("/api/pipeline/start", response_class=HTMLResponse)
async def api_pipeline_start(
    request: Request,
    video_path: str = Form(...),
    steps: list[str] = Form(default=[]),
) -> HTMLResponse:
    if not steps:
        steps = [name for name, _ in STEP_DEFS]
    cfg = get_config()
    vp = Path(video_path)
    if not vp.exists():
        logger.warning("/api/pipeline/start rejected — file not found: %s", vp)
        return HTMLResponse(f'<p class="text-error">File not found: {vp}</p>', status_code=400)
    try:
        job_id = start_job(str(vp), set(steps), cfg)
    except ConcurrencyRejected as exc:
        return HTMLResponse(f'<p class="text-error">{exc}</p>', status_code=409)
    logger.info("/api/pipeline/start — accepted job_id=%s", job_id)
    return build_start_response(request, cfg, vp, request.cookies.get("active_film", ""))


@router.post("/api/pipeline/cancel/{job_id}", response_class=HTMLResponse)
async def api_pipeline_cancel(request: Request, job_id: str) -> HTMLResponse:
    """Cooperatively cancel a running job."""
    ok = cancel_job(job_id)
    job = get_job(job_id)
    if job is None:
        return HTMLResponse('<p class="text-error">Job not found.</p>', status_code=404)
    if not ok:
        return HTMLResponse('<p class="text-muted">Job already finished.</p>', status_code=409)
    from api.services.processing_service import enrich_jobs  # noqa: PLC0415

    enrich_jobs([job])
    return templates.TemplateResponse(
        request, "partials/processing_job.html", make_ctx(request, job=job)
    )


@router.get("/api/pipeline/job-card/{job_id}", response_class=HTMLResponse)
async def api_pipeline_job_card(request: Request, job_id: str) -> HTMLResponse:
    """Return the full .p-active card for polling-based outer-card refresh."""
    job = get_job(job_id)
    if job is None:
        return HTMLResponse('<p class="text-error">Job not found.</p>', status_code=404)
    from api.services.processing_service import enrich_jobs  # noqa: PLC0415

    enrich_jobs([job])
    return templates.TemplateResponse(
        request, "partials/processing_job.html", make_ctx(request, job=job)
    )


@router.get("/api/pipeline/stream/{job_id}")
async def api_pipeline_stream(request: Request, job_id: str) -> StreamingResponse:
    """Stream pipeline progress as Server-Sent Events.

    Event types: ``log`` (log row), ``update`` (stepper), ``done`` /
    ``error`` / ``cancelled`` (terminal — stream closes after exactly one).
    Unknown job_id → single ``error`` frame then close.
    """
    locale = request.cookies.get("locale", "pt_BR")

    async def generator():
        job = get_job(job_id)
        if not job:
            yield "event: error\ndata: <p class='text-error'>Job not found.</p>\n\n"
            return

        sub = job.subscribe()
        logger.info("[job=%s] SSE connected (subs=%d)", job.id, job.broadcaster.subscriber_count())
        try:
            for row in list(job.log):
                yield f"event: log\ndata: {render_log_row(row, locale)}\n\n"
            if job.status not in _TERMINAL_EVENTS:
                yield f"event: update\ndata: {render_stepper(job, locale)}\n\n"
            while True:
                try:
                    name, data = sub.get_nowait()
                except queue.Empty:
                    if job.status in _TERMINAL_EVENTS:
                        yield f"event: {job.status}\ndata: {render_stepper(job, locale)}\n\n"
                        return
                    await asyncio.sleep(0.4)
                    yield ": keepalive\n\n"
                    continue
                if name in _TERMINAL_EVENTS:
                    yield f"event: {name}\ndata: {render_stepper(job, locale)}\n\n"
                    return
                if name == "log":
                    yield f"event: log\ndata: {render_log_row(data, locale)}\n\n"
                    continue
                yield f"event: update\ndata: {render_stepper(job, locale)}\n\n"
        finally:
            job.unsubscribe(sub)
            logger.info("[job=%s] SSE disconnected", job.id)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
