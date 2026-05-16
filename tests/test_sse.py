"""
tests/test_sse.py
~~~~~~~~~~~~~~~~~~
Phase 1d of the FastAPI regression-recovery effort: SSE completion
semantics.

These tests pin the *event-emission and close contract* of the pipeline
stream:

  * progress frames MUST be typed ``event: update``
  * the stream MUST end with exactly one terminal typed frame —
    ``event: done`` on success or ``event: error`` on failure — and the
    generator MUST then stop (no further frames, no infinite loop)
  * the terminal frame MUST carry the final rendered stepper so the UI
    shows the done/error state
  * the client-side contract (``processing_job.html`` ``sse-swap`` /
    ``sse-close`` event names) MUST match what the server emits, and the
    vendored ``htmx-ext-sse.js`` MUST split ``sse-close`` on ``,`` so a
    multi-name ``sse-close="done,error"`` registers one listener per
    event name (not a single literal ``"done,error"`` listener that
    never fires).

All tests are hermetic: no real pipeline / CLIP / video. The job's
``events`` queue and status are driven directly so the emitted sequence
is fully controlled. ``api.routes.processing.get_job`` is monkeypatched
so the stream endpoint sees our hand-built job.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# sys.path bootstrap lives in tests/conftest.py. ``sse_client`` is a
# deliberately lighter fixture than the shared ``client`` (the stream
# generator never touches config or the data dir) and does NOT depend
# on conftest's ``tmp_config``/``client`` — conftest is not in this
# fixture's path. ``sse_client`` SELF-RESETS the job registry via its
# own ``jobs._registry.reset()``; that line is the only thing keeping
# these tests hermetic, so do not delete it on the assumption conftest
# already cleared the registry (it does not run for this fixture).


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def sse_client(monkeypatch):
    """TestClient + a hand-built job whose events queue we drive.

    The stream generator only needs ``get_job`` and ``_render_stepper``;
    it never touches config or the data directory, so this fixture is
    deliberately lighter than the ``client`` fixture in
    test_web_routes.py.
    """
    import api.jobs as jobs

    jobs._registry.reset()

    job = jobs.JobState(
        id="ssejob",
        video_path="data/raw/jeca_tatu.mp4",
        steps=[
            jobs.StepInfo(name=name, label=label)
            for name, label in jobs.STEP_DEFS
        ],
    )
    jobs._registry.add(job)

    import api.routes.processing as processing

    monkeypatch.setattr(processing, "get_job", jobs.get_job)

    from api.server import app

    with TestClient(app) as c:
        c.cookies.set("locale", "en")
        yield c, job


def _frames(raw: str) -> list[dict[str, str]]:
    """Parse an SSE byte stream into a list of {event, data} dicts.

    Comment lines (``: keepalive``) and blank separators are ignored.
    ``data:`` lines within one event are joined with newlines per the
    SSE spec.
    """
    frames: list[dict[str, str]] = []
    cur_event: str | None = None
    cur_data: list[str] = []
    for line in raw.split("\n"):
        line = line.rstrip("\r")
        if line == "":
            if cur_event is not None or cur_data:
                frames.append(
                    {
                        "event": cur_event or "message",
                        "data": "\n".join(cur_data),
                    }
                )
            cur_event = None
            cur_data = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            cur_event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            cur_data.append(line[len("data:"):].lstrip())
    if cur_event is not None or cur_data:
        frames.append(
            {"event": cur_event or "message", "data": "\n".join(cur_data)}
        )
    return frames


# ── Server: typed update / done / error frames ────────────────────────────────


def test_stream_emits_typed_update_then_single_done(sse_client):
    """Progress frames are ``event: update``; the stream ends with
    exactly one ``event: done`` frame and then the generator stops."""
    client, job = sse_client

    # Drive the queue as the runner would: two progress signals then a
    # terminal "done", with status flipped to mirror the real runner.
    job.steps[0].state = "active"
    job.events.put("update")
    job.steps[0].state = "done"
    job.steps[1].state = "active"
    job.progress = 0.4
    job.events.put("update")
    job.status = "done"
    job.progress = 1.0
    for s in job.steps:
        if s.state not in ("done", "skipped", "error"):
            s.state = "done"
    job.events.put("done")

    with client.stream("GET", f"/api/pipeline/stream/{job.id}") as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        body = "".join(chunk for chunk in r.iter_text())

    frames = _frames(body)
    assert frames, f"no frames parsed from: {body!r}"

    events = [f["event"] for f in frames]
    assert events.count("done") == 1, f"expected exactly one done: {events}"
    assert "error" not in events, f"unexpected error frame: {events}"
    assert "message" not in events, (
        f"generic untyped 'message' frame emitted — server must type "
        f"every frame: {events}"
    )
    # The terminal frame is the LAST frame (generator stops after it).
    assert events[-1] == "done", f"done must be the final frame: {events}"
    # All non-terminal frames are progress updates.
    assert all(e == "update" for e in events[:-1]), events

    done_frame = frames[-1]
    assert "processing-done" in done_frame["data"], done_frame["data"]
    assert "pipeline-steps" in done_frame["data"]


def test_stream_emits_single_error_terminal_frame(sse_client):
    """A failed job ends the stream with exactly one ``event: error``
    frame carrying the error stepper, then stops."""
    client, job = sse_client

    job.steps[0].state = "active"
    job.events.put("update")
    job.steps[0].state = "error"
    job.error_msg = "boom: model not found"
    job.status = "error"
    job.events.put("error")

    with client.stream("GET", f"/api/pipeline/stream/{job.id}") as r:
        assert r.status_code == 200
        body = "".join(chunk for chunk in r.iter_text())

    frames = _frames(body)
    events = [f["event"] for f in frames]
    assert events.count("error") == 1, events
    assert "done" not in events, events
    assert "message" not in events, events
    assert events[-1] == "error", f"error must be the final frame: {events}"

    err_frame = frames[-1]
    assert "processing-error" in err_frame["data"], err_frame["data"]
    assert "boom: model not found" in err_frame["data"]


def test_stream_emits_single_cancelled_terminal_frame(sse_client):
    """A cancelled job ends the stream with exactly one ``event:
    cancelled`` frame, then stops — same close contract as done/error
    so the EventSource does not reconnect (Phase 4)."""
    client, job = sse_client

    job.steps[0].state = "active"
    job.events.put("update")
    job.steps[0].state = "error"
    job.status = "cancelled"
    job.error_msg = "Cancelled by user."
    job.events.put("cancelled")

    with client.stream("GET", f"/api/pipeline/stream/{job.id}") as r:
        assert r.status_code == 200
        body = "".join(chunk for chunk in r.iter_text())

    frames = _frames(body)
    events = [f["event"] for f in frames]
    assert events.count("cancelled") == 1, events
    assert "done" not in events and "error" not in events, events
    assert "message" not in events, events
    assert events[-1] == "cancelled", f"cancelled must be final: {events}"
    assert "processing-cancelled" in frames[-1]["data"], frames[-1]["data"]


def test_stream_terminal_status_without_queued_signal_closes(sse_client):
    """Defensive path: if status is terminal (``cancelled``) but no
    terminal signal was queued, the stream still emits exactly one
    matching terminal frame and stops (no infinite keepalive loop)."""
    client, job = sse_client
    job.status = "cancelled"  # no events queued at all

    with client.stream("GET", f"/api/pipeline/stream/{job.id}") as r:
        assert r.status_code == 200
        body = "".join(chunk for chunk in r.iter_text())

    frames = _frames(body)
    events = [f["event"] for f in frames]
    assert events == ["cancelled"], events


def test_stream_job_not_found_is_typed_error(sse_client):
    """An unknown job id yields a single typed ``error`` frame and
    closes (the client must be able to stop on it, not reconnect)."""
    client, _ = sse_client
    with client.stream("GET", "/api/pipeline/stream/nope") as r:
        assert r.status_code == 200
        body = "".join(chunk for chunk in r.iter_text())
    frames = _frames(body)
    events = [f["event"] for f in frames]
    assert events == ["error"], events
    assert "Job not found" in frames[0]["data"]


# ── Client contract: template + vendored JS split fix ─────────────────────────

REPO = Path(__file__).parent.parent


def test_processing_job_template_sse_attrs_match_server_events():
    """``processing_job.html`` must listen for the event names the
    server actually emits: ``sse-swap`` on the progress event and
    ``sse-close`` on the terminal events."""
    html = (REPO / "web/templates/partials/processing_job.html").read_text()
    # Phase 4 extended the terminal set with ``cancelled`` (a cancelled
    # job is terminal and must close the stream exactly like done/error).
    m_close = re.search(r'sse-close="([^"]+)"', html)
    assert m_close, "no sse-close attribute in processing_job.html"
    close_names = {n.strip() for n in m_close.group(1).split(",")}
    assert {"done", "error", "cancelled"} <= close_names, close_names
    # sse-swap must include "update" (the progress event the server now
    # emits) AND every terminal event so the final stepper is shown
    # before the source closes.
    m = re.search(r'sse-swap="([^"]+)"', html)
    assert m, "no sse-swap attribute in processing_job.html"
    swap_names = {n.strip() for n in m.group(1).split(",")}
    assert "update" in swap_names, swap_names
    assert {"done", "error", "cancelled"} <= swap_names, (
        f"sse-swap must also listen to done/error/cancelled so the "
        f"terminal stepper is rendered before close: {swap_names}"
    )
    assert "message" not in swap_names, (
        "sse-swap still listens on the generic 'message' event — the "
        "server no longer emits untyped frames"
    )


def test_vendored_sse_ext_splits_sse_close_on_comma():
    """The vendored ``htmx-ext-sse.js`` must split ``sse-close`` on ``,``
    and register a listener per event name.

    Before the Phase 1d fix it did
    ``source.addEventListener(closeAttribute, ...)`` with the RAW
    attribute, so ``sse-close="done,error"`` registered a single
    listener for an event literally named ``"done,error"`` which the
    server never emits — the EventSource then reconnected forever after
    the job finished.

    No JS runtime is available in CI, so this guards the fix at the
    source level: the close-handling block must split the attribute
    (``.split(',')``) and iterate, mirroring the existing ``sse-swap``
    handling. This plus the matching template contract above is the
    verification basis; live browser behaviour is re-checked manually in
    Phase 8.
    """
    js = (REPO / "web/static/js/htmx-ext-sse.js").read_text()
    assert "closeAttribute.split(','" in js or 'closeAttribute.split(","' in js, (
        "sse-close is not split on comma — multi-name sse-close will "
        "never fire and the EventSource will reconnect after the job ends"
    )
    # The single-literal-listener anti-pattern must be gone.
    assert "source.addEventListener(closeAttribute," not in js, (
        "raw closeAttribute is still passed to addEventListener — the "
        "comma-joined literal listener bug is not fixed"
    )
    # The close handler must register a listener per split name (mirrors
    # the existing sse-swap loop): a per-name addEventListener inside the
    # split loop.
    close_region = js[js.index("var closeAttribute"):]
    assert "closeEventNames" in close_region
    assert "source.addEventListener(closeEventName," in close_region
