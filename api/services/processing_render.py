"""Processing tab — rendering helpers and context builder.

Extracted from ``api/routes/processing.py`` (A2 route-thinning refactor).
The SSE generator in the route calls ``render_stepper`` / ``render_log_row``
directly; ``build_processing_context`` is shared with ``api/server.py``'s
``_TAB_CONTEXT_BUILDERS["processing"]`` so both the HTMX fragment path
and the full-page path use the same context.
"""

from __future__ import annotations

from pathlib import Path

from api.deps import _get_translations, get_config, make_ctx
from api.jobs import STEP_DEFS, JobState, active_jobs
from api.services.chrome_service import build_chrome_context
from api.templates import templates


# ── Rendering helpers ─────────────────────────────────────────────────────────


def render_stepper(job: JobState, locale: str = "pt_BR") -> str:
    """Render the stepper HTML fragment for SSE."""
    trans = _get_translations(locale)
    html = templates.env.get_template("partials/processing_stepper.html").render(
        job=job,
        _=trans.gettext,
    )
    return html.replace("\n", " ").strip()


def render_log_row(row: dict, locale: str = "pt_BR") -> str:
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


# ── Context builder ───────────────────────────────────────────────────────────


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
      * ``gpu_metrics`` — optional CPU/RAM/VRAM resource metrics gated by
        ``cfg.proc.gpu_metrics_enabled``.
      * Each enriched ``JobState`` carries display fields
        (``film_title``, ``started_at_display``, ``elapsed_display``,
        ``active_step_idx``, …) the .p-active card header reads.
    """
    cfg = get_config()
    from api.services.processing_service import (
        aggregate_stats,
        build_active_step,
        build_job_queue,
        build_resource_metrics,
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
    from api.jobs import get_all_jobs  # noqa: PLC0415 - service-layer access

    active_step = build_active_step(jobs)
    metrics_enabled = bool(getattr(getattr(cfg, "proc", None), "gpu_metrics_enabled", False))
    gpu_metrics = build_resource_metrics() if metrics_enabled and active_step else []

    return {
        "films": films,
        "step_defs": STEP_DEFS,
        "jobs": jobs,
        "initial_log_lines": initial_log_lines,
        "stats": aggregate_stats(library_dir),
        "job_queue": build_job_queue(get_all_jobs()),
        "active_step": active_step,
        "gpu_metrics": gpu_metrics,
        "cfg": cfg,
    }


# ── Start-response builder ─────────────────────────────────────────────────────


def _derive_slug(cfg, video_path: Path, fallback: str) -> str:
    """Scan the library and derive the slug for the film at ``video_path``.

    Returns ``fallback`` (the active_film cookie value) when no match is found.
    """
    from cinemateca.library import scan_library  # noqa: PLC0415

    library_dir = Path(cfg.paths.library_dir)
    films = scan_library(library_dir)
    vp_resolved = video_path.resolve()
    new_slug = fallback
    for film in films:
        try:
            if film.raw_path.resolve() == vp_resolved:
                return film.slug
            if film.raw_path.name == vp_resolved.name:
                new_slug = film.slug
        except (OSError, RuntimeError):
            if film.raw_path.name == vp_resolved.name:
                new_slug = film.slug
    return new_slug


def build_start_response(request, cfg, video_path: Path, cookie_slug: str):
    """Compose the full HTML response for a successful pipeline start.

    Derives the slug for the film at ``video_path`` from the library
    (falling back to ``cookie_slug`` when no match is found), then returns
    an HTMLResponse carrying:
      - primary swap: the full Processing tab re-rendered for the new job
      - OOB swap: the left-pane film list with the new active film highlighted
      - cookie: active_film set to the derived slug
    """
    from fastapi.responses import HTMLResponse

    new_slug = _derive_slug(cfg, video_path, cookie_slug)

    # Primary swap: full Processing tab.
    proc_ctx = build_processing_context()
    tab_html = templates.env.get_template("partials/processing.html").render(
        make_ctx(request, active_film=new_slug, current_slug=new_slug, **proc_ctx)
    )

    # OOB swap: left-pane film list with the new active film highlighted
    chrome_ctx = build_chrome_context(cfg, current_slug=new_slug)
    lp_payload: dict = dict(chrome_ctx)
    lp_payload.update({"active_film": new_slug, "current_slug": new_slug})
    lp_ctx = make_ctx(request, **lp_payload)
    lp_html = templates.env.get_template("partials/_left_pane_body.html").render(lp_ctx)
    oob = f'<div id="lp-scroll" hx-swap-oob="innerHTML">{lp_html}</div>'

    response = HTMLResponse(tab_html + oob)
    response.set_cookie(
        "active_film", new_slug, max_age=86400 * 365, httponly=False, samesite="lax"
    )
    return response
