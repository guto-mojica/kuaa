"""Behavior-preserving gate for the processing route thinning (A2/A9)."""

from __future__ import annotations

from tests._snapshot import assert_snapshot


def test_tab_processing_snapshot(client) -> None:
    r = client.get("/tab/processing")
    assert r.status_code == 200
    assert_snapshot("processing/tab_processing_empty", r.text)


def test_stream_unknown_job_emits_single_error_frame(client) -> None:
    # Unknown job → exactly one terminal error frame, then close (the documented contract).
    with client.stream("GET", "/api/pipeline/stream/deadbeef") as resp:
        assert resp.status_code == 200
        body = "".join(chunk for chunk in resp.iter_text())
    assert body.count("event: error") == 1
    assert "Job not found" in body
