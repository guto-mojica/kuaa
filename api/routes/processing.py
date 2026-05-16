"""Processing tab routes — pipeline start, SSE stream, status."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from api.deps import get_config, make_ctx
from api.jobs import STEP_DEFS, JobState, active_jobs, get_job, start_job
from api.templates import templates

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Rendering helpers ─────────────────────────────────────────────────────────

def _render_stepper(job: JobState) -> str:
    """Render the stepper HTML fragment for SSE (no request context needed)."""
    html = templates.env.get_template("partials/processing_stepper.html").render(
        job=job,
    )
    return html.replace("\n", " ").strip()


# ── Routes ────────────────────────────────────────────────────────────────────

def build_processing_context() -> dict:
    """Build the template context the processing tab partial needs.

    Shared by the ``/tab/processing`` HTMX fragment and the
    ``/processing`` full-page route so both render the step checklist
    and active-job list identically.
    """
    cfg = get_config()
    from cinemateca.library import scan_library

    films = scan_library(
        raw_dir=Path(cfg.paths.raw_dir),
        metadata_dir=Path(cfg.paths.metadata_dir),
    )
    jobs = active_jobs()

    return {"films": films, "step_defs": STEP_DEFS, "jobs": jobs}


@router.get("/tab/processing", response_class=HTMLResponse)
async def tab_processing(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "partials/processing.html",
        make_ctx(request, **build_processing_context()),
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
        return HTMLResponse(
            f'<p class="text-error">File not found: {vp}</p>', status_code=400
        )

    job_id = start_job(str(vp), set(steps), cfg)
    job = get_job(job_id)

    return templates.TemplateResponse(
        request,
        "partials/processing_job.html",
        make_ctx(request, job=job),
    )


@router.get("/api/pipeline/stream/{job_id}")
async def api_pipeline_stream(job_id: str) -> StreamingResponse:
    async def generator():
        job = get_job(job_id)
        if not job:
            # Typed terminal frame so the client closes (and does NOT
            # reconnect) on an unknown job id.
            yield "event: error\ndata: <p class='text-error'>Job not found.</p>\n\n"
            return

        while True:
            try:
                signal = job.events.get_nowait()
            except Exception:
                # Queue empty — wait, send keepalive
                if job.status in ("done", "error"):
                    # Status flipped without a queued terminal signal
                    # (defensive): emit the matching terminal frame and
                    # stop so the stream never loops forever.
                    event = "done" if job.status == "done" else "error"
                    yield f"event: {event}\ndata: {_render_stepper(job)}\n\n"
                    return
                await asyncio.sleep(0.4)
                yield ": keepalive\n\n"
                continue

            html = _render_stepper(job)

            if signal in ("done", "error"):
                # Exactly one terminal typed frame carrying the final
                # rendered stepper, then close the stream.
                yield f"event: {signal}\ndata: {html}\n\n"
                return

            # Progress / intermediate frame.
            yield f"event: update\ndata: {html}\n\n"

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
