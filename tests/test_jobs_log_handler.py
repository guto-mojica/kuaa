"""Tests for the contextual pipeline log handler.

The handler captures records from the pipeline loggers (``kuaa.*``
and ``api.jobs``) while a job is active, appends them to the job's
ring buffer, AND publishes a ``log`` event on the broadcaster so live
SSE consumers see each row in real time.

Lifecycle: installed by :func:`api.jobs.install_pipeline_log_handler`
inside a context manager so the runner's ``finally`` block guarantees
removal even on crash. Two simultaneous active jobs would each install
their own handler with their own filter (job_id propagated via the
log record's ``extra``); but the single-global-active-job policy means
in practice there is one handler at a time.
"""

from __future__ import annotations

import logging
import queue

import pytest


def test_log_handler_appends_to_job_log_and_publishes_event():
    """Captured records land in BOTH the durable ring buffer and the
    live broadcaster.

    The buffer is what a returning user replays; the broadcast event
    is what a live consumer's SSE stream picks up. Both paths must
    fire for every captured record so the two layers stay in sync.
    """
    from api.jobs import JobState, install_pipeline_log_handler

    job = JobState(id="lh1", video_path="x.mp4")
    probe = job.subscribe()
    pipeline_logger = logging.getLogger("kuaa.pipeline")

    with install_pipeline_log_handler(job):
        pipeline_logger.info("extracted 412 keyframes")

    # Durable: the buffer recorded the row with template-ready keys.
    assert len(job.log) == 1
    row = job.log[0]
    assert row["m"] == "extracted 412 keyframes"
    assert row["lv"] == "i"
    assert isinstance(row["t"], str) and len(row["t"]) == 8  # HH:MM:SS

    # Live: the broadcaster published exactly one ("log", row) event.
    name, data = probe.get_nowait()
    assert name == "log"
    assert data == row
    with pytest.raises(queue.Empty):
        probe.get_nowait()


def test_log_handler_maps_python_levels_to_template_codes():
    """Python ``levelno`` codes MUST collapse onto the template
    vocabulary (``i|d|w|s|e``) that ``processing_log_line.html``
    knows how to style.
    """
    from api.jobs import JobState, install_pipeline_log_handler

    job = JobState(id="lh2", video_path="x.mp4")
    lg = logging.getLogger("kuaa.test_levels")

    with install_pipeline_log_handler(job):
        lg.debug("d row")
        lg.info("i row")
        lg.warning("w row")
        lg.error("e row")
        lg.critical("e row crit")

    rows = list(job.log)
    assert [r["lv"] for r in rows] == ["d", "i", "w", "e", "e"]


def test_log_handler_is_removed_on_context_exit():
    """After the context manager exits, subsequent log records MUST
    NOT land in the job buffer.

    Otherwise a long-running webserver would accumulate every log
    record from every finished job's source into the most recently
    installed handler — handlers must be scoped to job lifetime.
    """
    from api.jobs import JobState, install_pipeline_log_handler

    job = JobState(id="lh3", video_path="x.mp4")
    lg = logging.getLogger("kuaa.pipeline")

    with install_pipeline_log_handler(job):
        lg.info("inside")
    lg.info("after — must be ignored")

    rows = list(job.log)
    msgs = [r["m"] for r in rows]
    assert msgs == ["inside"], msgs


def test_log_handler_ignores_records_from_unrelated_loggers():
    """The handler attaches to the pipeline-logger namespace ONLY —
    a totally unrelated logger (httpx, uvicorn) MUST NOT pollute the
    job log.
    """
    from api.jobs import JobState, install_pipeline_log_handler

    job = JobState(id="lh4", video_path="x.mp4")
    pipeline = logging.getLogger("kuaa.pipeline")
    unrelated = logging.getLogger("httpx._client")

    with install_pipeline_log_handler(job):
        pipeline.info("pipeline row")
        unrelated.info("http row")

    msgs = [r["m"] for r in job.log]
    assert msgs == ["pipeline row"]


def test_log_handler_removes_on_exception():
    """A crash inside the ``with`` block MUST still remove the
    handler (no leaked handlers across jobs).
    """
    from api.jobs import JobState, install_pipeline_log_handler

    job = JobState(id="lh5", video_path="x.mp4")
    lg = logging.getLogger("kuaa.pipeline")

    with pytest.raises(RuntimeError):
        with install_pipeline_log_handler(job):
            raise RuntimeError("kaboom")
    lg.info("after crash — must NOT land")

    assert list(job.log) == []
