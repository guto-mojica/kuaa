"""Processing tab routes — pipeline start, SSE stream, status.

T9: ``/tab/processing`` accepts an optional ``?film=<slug>`` query
parameter (wired for completeness; the processing tab itself always
shows the global job queue, not per-film-filtered jobs).
"""

from __future__ import annotations

import asyncio
import logging
import queue
from pathlib import Path

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from api.deps import _get_translations, film_slug_query, get_config, make_ctx
from api.jobs import (
    STEP_DEFS,
    ConcurrencyRejected,
    JobState,
    active_jobs,
    cancel_job,
    get_job,
    start_job,
)
from api.services.chrome_service import build_chrome_context
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


def _render_stepper(job: JobState, locale: str = "pt_BR") -> str:
    """Render the stepper HTML fragment for SSE."""
    trans = _get_translations(locale)
    html = templates.env.get_template("partials/processing_stepper.html").render(
        job=job,
        _=trans.gettext,
    )
    return html.replace("\n", " ").strip()


def _render_log_row(row: dict, locale: str = "pt_BR") -> str:
    """Render one ``processing_log_line.html`` row as a single-line
    SSE payload.

    ``row`` is the dict shape produced by the pipeline log handler
    and stored in ``JobState.log`` — keys ``t`` (timestamp HH:MM:SS),
    ``lv`` (one of ``i|d|w|s|e``), ``m`` (message text).
    """
    trans = _get_translations(locale)
    html = templates.env.get_template("partials/processing_log_line.html").render(
        line=row,
        _=trans.gettext,
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
    active = active_jobs()
    jobs = enrich_jobs(active)

    # Seed the terminal-log pane with the active job's captured log
    # history. This is what makes "user leaves and returns mid-job"
    # paint with the full pipeline transcript before the SSE
    # connection even establishes. The job's ``log`` is a bounded
    # deque; copy to a list so the template doesn't mutate it.
    # Single-global-active-job means at most one job contributes.
    initial_log_lines: list[dict] = []
    if active:
        initial_log_lines = list(active[0].log)

    # ``job_queue`` reads the registry's *full* recent history (terminal
    # + active), not just the currently running set.
    from api.jobs import _registry  # noqa: PLC0415 - service-layer access

    return {
        "films": films,
        "step_defs": STEP_DEFS,
        "jobs": jobs,
        "initial_log_lines": initial_log_lines,
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
    ctx = build_processing_context()
    logger.info(
        "/tab/processing — slug=%s active_jobs=%d initial_log_lines=%d",
        slug,
        len(ctx["jobs"]),
        len(ctx["initial_log_lines"]),
    )
    return templates.TemplateResponse(
        request,
        "partials/processing.html",
        make_ctx(request, current_slug=slug, **ctx),
    )


@router.post("/api/pipeline/start", response_class=HTMLResponse)
async def api_pipeline_start(
    request: Request,
    video_path: str = Form(...),
    steps: list[str] = Form(default=[]),
) -> HTMLResponse:
    if not steps:
        steps = [name for name, _ in STEP_DEFS]
    logger.info(
        "/api/pipeline/start — video_path=%s steps=%s",
        video_path, sorted(steps),
    )

    cfg = get_config()
    vp = Path(video_path)
    if not vp.exists():
        logger.warning(
            "/api/pipeline/start rejected — file not found: %s", vp,
        )
        return HTMLResponse(f'<p class="text-error">File not found: {vp}</p>', status_code=400)

    try:
        job_id = start_job(str(vp), set(steps), cfg)
    except ConcurrencyRejected as exc:
        logger.info(
            "/api/pipeline/start rejected — already-running job=%s",
            getattr(exc.active, "id", "?"),
        )
        return HTMLResponse(f'<p class="text-error">{exc}</p>', status_code=409)
    job = get_job(job_id)
    logger.info("/api/pipeline/start — accepted job_id=%s", job_id)

    if job is not None:
        from api.services.processing_service import enrich_jobs  # noqa: PLC0415

        enrich_jobs([job])

    # Derive the slug for the film being processed so we can update
    # active_film cookie and refresh the left-pane film list in one shot.
    from cinemateca.library import scan_library  # noqa: PLC0415

    library_dir = Path(cfg.paths.library_dir)
    films = scan_library(library_dir)
    vp_resolved = vp.resolve()
    new_slug = request.cookies.get("active_film", "")
    for film in films:
        try:
            if film.raw_path.resolve() == vp_resolved:
                new_slug = film.slug
                break
            if film.raw_path.name == vp_resolved.name:
                new_slug = film.slug
                break
        except (OSError, RuntimeError):
            if film.raw_path.name == vp_resolved.name:
                new_slug = film.slug
                break

    # Primary swap: job card for #processing-job
    job_html = templates.env.get_template("partials/processing_job.html").render(
        make_ctx(request, job=job, active_film=new_slug, current_slug=new_slug)
    )

    # OOB swap: left-pane film list with the new active film highlighted
    chrome_ctx = build_chrome_context(cfg, current_slug=new_slug)
    lp_ctx = make_ctx(request, **chrome_ctx, active_film=new_slug, current_slug=new_slug)
    lp_html = templates.env.get_template("partials/_left_pane_body.html").render(lp_ctx)
    oob = f'<div id="lp-scroll" hx-swap-oob="innerHTML">{lp_html}</div>'

    response = HTMLResponse(job_html + oob)
    response.set_cookie("active_film", new_slug, max_age=86400 * 365, httponly=False, samesite="lax")
    return response


@router.post("/api/pipeline/cancel/{job_id}", response_class=HTMLResponse)
async def api_pipeline_cancel(request: Request, job_id: str) -> HTMLResponse:
    """Cooperatively cancel a running job.

    The runner polls the cancel flag between steps and finalizes the
    job as ``cancelled`` (a terminal SSE frame then closes the stream).
    """
    logger.info("/api/pipeline/cancel — job_id=%s", job_id)
    ok = cancel_job(job_id)
    job = get_job(job_id)
    if job is None:
        logger.warning("/api/pipeline/cancel — unknown job_id=%s", job_id)
        return HTMLResponse('<p class="text-error">Job not found.</p>', status_code=404)
    if not ok:
        logger.info(
            "/api/pipeline/cancel — already terminal job_id=%s status=%s",
            job_id, job.status,
        )
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


@router.get("/api/pipeline/job-card/{job_id}", response_class=HTMLResponse)
async def api_pipeline_job_card(request: Request, job_id: str) -> HTMLResponse:
    """Return the full .p-active card for polling-based outer-card refresh.

    The ``processing_job.html`` template includes ``hx-trigger="every 3s"``
    on the article when the job is active; this endpoint serves the refresh.
    Polling stops automatically when the job reaches a terminal state because
    the returned article omits the ``hx-trigger`` attribute.
    """
    job = get_job(job_id)
    if job is None:
        return HTMLResponse('<p class="text-error">Job not found.</p>', status_code=404)

    from api.services.processing_service import enrich_jobs  # noqa: PLC0415

    enrich_jobs([job])
    return templates.TemplateResponse(
        request,
        "partials/processing_job.html",
        make_ctx(request, job=job),
    )


@router.get("/api/pipeline/stream/{job_id}")
async def api_pipeline_stream(request: Request, job_id: str) -> StreamingResponse:
    """Stream pipeline progress for a job as Server-Sent Events.

    Event vocabulary:

      * ``event: log``    — one captured pipeline log row (rendered as
                            ``processing_log_line.html``). Replayed
                            from ``job.log`` on connect so a returning
                            consumer sees the full history; then
                            streamed live from the broadcaster.
      * ``event: update`` — stepper progress signal. The data payload
                            is the current ``processing_stepper.html``
                            so the consumer always sees fresh state.
      * ``event: done`` / ``event: error`` / ``event: cancelled`` —
                            terminal: exactly one such frame, then the
                            generator returns and the stream closes.

    Multi-consumer safe: the underlying ``EventBroadcaster`` fans
    events out to every subscriber, so two browser tabs (or a page
    reload during a job) both see the live stream without racing for
    queue entries. The generator subscribes on entry and unsubscribes
    in a ``finally`` so a dropped connection doesn't leak a queue.

    An unknown ``job_id`` yields a single terminal ``event: error``
    frame and closes immediately.
    """
    locale = request.cookies.get("locale", "pt_BR")

    async def generator():
        job = get_job(job_id)
        if not job:
            logger.info("SSE stream for unknown job_id=%s — emitting error and closing", job_id)
            yield "event: error\ndata: <p class='text-error'>Job not found.</p>\n\n"
            return

        # Subscribe BEFORE replaying the buffer so an event published
        # in the gap between buffer drain and stream-loop entry lands
        # in our per-connection queue rather than being lost.
        sub = job.subscribe()
        sub_count = job.broadcaster.subscriber_count()
        logger.info(
            "[job=%s] SSE consumer connected (subscribers=%d, log_buffered=%d, status=%s)",
            job.id, sub_count, len(job.log), job.status,
        )

        try:
            # 1) Replay the full buffered log history so a late
            #    consumer (returning user, second tab) sees every line
            #    the pipeline emitted before they connected.
            for row in list(job.log):
                yield f"event: log\ndata: {_render_log_row(row, locale)}\n\n"

            # 2) Snapshot the current stepper as the first ``update``
            #    frame — but ONLY if the job is still live. A terminal
            #    job's first (and only) frame must be the matching
            #    terminal event, per the close contract; emitting an
            #    extra ``update`` first would lie about progress and
            #    confuse the test contract that pins ``frames ==
            #    [terminal]`` for the defensive-entry path.
            if job.status not in _TERMINAL_EVENTS:
                yield f"event: update\ndata: {_render_stepper(job, locale)}\n\n"

            # 3) Live loop: drain the per-connection subscriber queue.
            while True:
                try:
                    name, data = sub.get_nowait()
                except queue.Empty:
                    if job.status in _TERMINAL_EVENTS:
                        # Defensive close: status flipped without a
                        # queued terminal signal (or the signal was
                        # consumed while we were in the replay block).
                        # Either way, emit the matching terminal frame
                        # exactly once so the client closes cleanly.
                        yield f"event: {job.status}\ndata: {_render_stepper(job, locale)}\n\n"
                        return
                    await asyncio.sleep(0.4)
                    yield ": keepalive\n\n"
                    continue

                if name in _TERMINAL_EVENTS:
                    yield f"event: {name}\ndata: {_render_stepper(job, locale)}\n\n"
                    return
                if name == "log":
                    yield f"event: log\ndata: {_render_log_row(data, locale)}\n\n"
                    continue
                # default: progress / update — render current stepper
                yield f"event: update\ndata: {_render_stepper(job, locale)}\n\n"
        finally:
            job.unsubscribe(sub)
            logger.info(
                "[job=%s] SSE consumer disconnected (subscribers=%d)",
                job.id, job.broadcaster.subscriber_count(),
            )

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
