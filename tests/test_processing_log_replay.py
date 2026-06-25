"""Tests for the build_processing_context → initial_log_lines hand-off.

When a user navigates back to ``/tab/processing`` mid-pipeline, the
page MUST render with the buffered log history before SSE even
connects. This pins that contract: the context populated by
:func:`api.routes.processing.build_processing_context` carries the
active job's full ``log`` deque as ``initial_log_lines`` so the
template's ``{% for line in initial_log_lines %}`` block paints them
on the first frame.
"""

from __future__ import annotations

import pytest


@pytest.fixture()
def reset_registry():
    import api.jobs as jobs

    jobs._registry.reset()
    yield
    jobs._registry.reset()


def test_initial_log_lines_seeded_from_active_job_buffer(reset_registry, monkeypatch):
    """An active job's ``log`` rows MUST appear in the context as
    ``initial_log_lines`` so the page paints history before SSE.
    """
    import api.jobs as jobs
    from api.routes.processing import build_processing_context

    job = jobs.JobState(
        id="ctxjob",
        video_path="x.mp4",
        steps=[jobs.StepInfo(name=name, label=label) for name, label in jobs.STEP_DEFS],
        status=jobs.STATUS_RUNNING,
    )
    job.log.append({"t": "00:00:01", "lv": "i", "m": "alpha"})
    job.log.append({"t": "00:00:02", "lv": "s", "m": "beta"})
    jobs._registry.add(job)

    # build_processing_context calls into config + scan_library; stub
    # the library scan so this test stays hermetic.
    monkeypatch.setattr("kuaa.library.scan_library", lambda library_dir: [])
    monkeypatch.setattr("api.services.processing_service.aggregate_stats", lambda lib: {})
    monkeypatch.setattr("api.services.processing_service.build_job_queue", lambda *a, **kw: [])
    monkeypatch.setattr("api.services.processing_service.build_active_step", lambda jobs_: None)

    ctx = build_processing_context()

    assert "initial_log_lines" in ctx
    lines = ctx["initial_log_lines"]
    msgs = [row["m"] for row in lines]
    assert msgs == ["alpha", "beta"], msgs


def test_initial_log_lines_empty_when_no_active_job(reset_registry, monkeypatch):
    """With no active jobs, initial_log_lines MUST be an empty list
    so the template renders the 'Waiting for events…' empty-state.
    """
    from api.routes.processing import build_processing_context

    monkeypatch.setattr("kuaa.library.scan_library", lambda library_dir: [])
    monkeypatch.setattr("api.services.processing_service.aggregate_stats", lambda lib: {})
    monkeypatch.setattr("api.services.processing_service.build_job_queue", lambda *a, **kw: [])
    monkeypatch.setattr("api.services.processing_service.build_active_step", lambda jobs_: None)

    ctx = build_processing_context()

    assert ctx["initial_log_lines"] == []


def test_tab_processing_renders_buffered_log_rows_and_sse_wiring(client, reset_registry):
    """End-to-end: GET /tab/processing for an active job with buffered
    log MUST render the rows AND wire #proc-log for SSE log events.

    This pins the chain that fixes 'leaves and returns blank':
      * buffered log → initial_log_lines (server-side replay)
      * .lines carries sse-swap='log' + hx-swap='beforeend' so live
        events append rather than replace
      * sse-close on #proc-log triggers EventSource.close() on terminal
    """
    import api.jobs as jobs

    job = jobs.JobState(
        id="rendjob",
        video_path="x.mp4",
        steps=[jobs.StepInfo(name=name, label=label) for name, label in jobs.STEP_DEFS],
        status=jobs.STATUS_RUNNING,
    )
    job.log.append({"t": "00:00:01", "lv": "i", "m": "captured row alpha"})
    job.log.append({"t": "00:00:02", "lv": "s", "m": "captured row beta"})
    jobs._registry.add(job)

    resp = client.get("/tab/processing")
    assert resp.status_code == 200
    body = resp.text

    # Buffered log rows replayed server-side BEFORE SSE even connects.
    assert "captured row alpha" in body
    assert "captured row beta" in body

    # SSE wiring on #proc-log + .lines (the contract this fix introduced).
    assert 'sse-connect="/api/pipeline/stream/rendjob"' in body
    assert 'sse-close="done,error,cancelled"' in body
    assert 'sse-swap="log"' in body
    assert 'hx-swap="beforeend"' in body

    # The previous "every 3s outerHTML" poll on the job card is gone —
    # the inner SSE is the live source of truth and the outer poll
    # caused the EventSource to churn every 3 seconds.
    assert 'hx-trigger="every 3s"' not in body
