"""Processing tab routes — pipeline start, SSE stream, status.

T9: ``/tab/processing`` accepts an optional ``?film=<slug>`` query
parameter (wired for completeness; the processing tab itself always
shows the global job queue, not per-film-filtered jobs).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from api.deps import film_slug_query, get_config, make_ctx
from api.jobs import (
    STEP_DEFS,
    ConcurrencyRejected,
    JobState,
    active_jobs,
    cancel_job,
    get_job,
    start_job,
)
from api.templates import templates

# Terminal SSE events: after one of these the generator emits exactly one
# typed frame carrying the final stepper, then returns (stream closes).
# ``cancelled`` joins ``done``/``error`` here so a cancelled job closes
# the stream the same way (no reconnect loop) — kept in sync with
# ``processing_job.html``'s ``sse-close`` / ``sse-swap`` attributes.
_TERMINAL_EVENTS = ("done", "error", "cancelled")

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

    Mojica Task 24 extends the context with the new ``.p-cp`` layout's
    requirements:

      * ``initial_log_lines`` — empty; SSE feeds the terminal log
        live. A future tail-of-buffer hook can seed this with the
        most recent N events.
      * ``stats`` — aggregate frames/scenes/embeddings/descriptions/
        faces/objects counts (see :func:`processing_service.aggregate_stats`).
      * ``job_queue`` — recent-job history mapped to the .p-queue
        item-status vocabulary.
      * ``active_step`` — sub-step detail for the right pane (.p-rp);
        ``None`` when no job is running so the partial omits the pane.
      * ``gpu_metrics`` — empty by default; gated by
        ``cfg.proc.gpu_metrics_enabled`` (off in shipped config).
      * Each enriched ``JobState`` carries display fields
        (``film_title``, ``started_at_display``, ``elapsed_display``,
        ``active_step_idx``, …) the .p-active card header reads.
    """
    cfg = get_config()
    from api.services.processing_service import (
        aggregate_stats,
        build_active_step,
        build_job_queue,
        enrich_jobs,
    )
    from cinemateca.library import scan_library

    library_dir = Path(cfg.paths.library_dir)
    films = scan_library(library_dir)
    jobs = enrich_jobs(active_jobs())

    # ``job_queue`` reads the registry's *full* recent history (terminal
    # + active), not just the currently running set.
    from api.jobs import _registry  # noqa: PLC0415 - service-layer access

    return {
        "films": films,
        "step_defs": STEP_DEFS,
        "jobs": jobs,
        "initial_log_lines": [],
        "stats": aggregate_stats(library_dir),
        "job_queue": build_job_queue(_registry.all()),
        "active_step": build_active_step(jobs),
        "gpu_metrics": [],
        "cfg": cfg,
    }


@router.get("/tab/processing", response_class=HTMLResponse)
async def tab_processing(
    request: Request,
    slug: str | None = Depends(film_slug_query),
) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "partials/processing.html",
        make_ctx(request, current_slug=slug, **build_processing_context()),
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
        return HTMLResponse(f'<p class="text-error">File not found: {vp}</p>', status_code=400)

    try:
        job_id = start_job(str(vp), set(steps), cfg)
    except ConcurrencyRejected as exc:
        # Single-global-active-job policy: surface a clear rejection
        # instead of launching a second pipeline.
        return HTMLResponse(f'<p class="text-error">{exc}</p>', status_code=409)
    job = get_job(job_id)

    # Enrich the freshly-started job with the display fields the
    # ``.p-active`` card header reads (film_title / started_at_display /
    # elapsed_display / active_step_idx). Without this the POST response
    # would render a sparse card whereas the GET path renders the rich
    # one.
    if job is not None:
        from api.services.processing_service import enrich_jobs  # noqa: PLC0415

        enrich_jobs([job])

    return templates.TemplateResponse(
        request,
        "partials/processing_job.html",
        make_ctx(request, job=job),
    )


@router.post("/api/pipeline/cancel/{job_id}", response_class=HTMLResponse)
async def api_pipeline_cancel(request: Request, job_id: str) -> HTMLResponse:
    """Cooperatively cancel a running job.

    The runner polls the cancel flag between steps and finalizes the
    job as ``cancelled`` (a terminal SSE frame then closes the stream).
    """
    ok = cancel_job(job_id)
    job = get_job(job_id)
    if job is None:
        return HTMLResponse('<p class="text-error">Job not found.</p>', status_code=404)
    if not ok:
        return HTMLResponse(
            '<p class="text-muted">Job already finished.</p>',
            status_code=409,
        )
    # Same enrichment as the start path so the .p-active card renders
    # with its display fields populated post-cancel too.
    from api.services.processing_service import enrich_jobs  # noqa: PLC0415

    enrich_jobs([job])
    return templates.TemplateResponse(
        request,
        "partials/processing_job.html",
        make_ctx(request, job=job),
    )


@router.get("/api/pipeline/stream/{job_id}")
async def api_pipeline_stream(job_id: str) -> StreamingResponse:
    """Stream pipeline progress for a job as Server-Sent Events.

    Emits ``event: update`` frames carrying the rendered stepper HTML for
    each progress signal, then exactly one terminal ``event: done`` or
    ``event: error`` frame carrying the final stepper HTML, after which the
    generator returns and the stream closes. An unknown ``job_id`` yields a
    single terminal ``event: error`` frame and closes immediately.
    """

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
                if job.status in _TERMINAL_EVENTS:
                    # Status flipped without a queued terminal signal
                    # (defensive): emit the matching terminal frame and
                    # stop so the stream never loops forever. ``status``
                    # maps 1:1 onto the terminal event name.
                    yield f"event: {job.status}\ndata: {_render_stepper(job)}\n\n"
                    return
                await asyncio.sleep(0.4)
                yield ": keepalive\n\n"
                continue

            html = _render_stepper(job)

            if signal in _TERMINAL_EVENTS:
                # Exactly one terminal typed frame carrying the final
                # rendered stepper, then close the stream (``done``,
                # ``error`` or ``cancelled``).
                yield f"event: {signal}\ndata: {html}\n\n"
                return

            # Progress / intermediate frame.
            yield f"event: update\ndata: {html}\n\n"

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
