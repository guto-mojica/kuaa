"""Request-ID + access-log middleware (F5)."""

from __future__ import annotations

import logging
import uuid


def test_response_carries_request_id(client):
    r = client.get("/api/search", params={"q": "ab", "top_k": 5})
    assert "x-request-id" in {k.lower() for k in r.headers}
    rid = r.headers["x-request-id"]
    assert uuid.UUID(rid)  # well-formed UUID generated server-side


def test_inbound_request_id_is_echoed(client):
    given = "11111111-2222-3333-4444-555555555555"
    r = client.get(
        "/api/search",
        params={"q": "ab", "top_k": 5},
        headers={"X-Request-ID": given},
    )
    assert r.headers["x-request-id"] == given


def test_access_log_line_emitted(client, caplog):
    with caplog.at_level(logging.INFO, logger="api.access"):
        client.get("/api/search", params={"q": "ab", "top_k": 5})
    lines = [r for r in caplog.records if r.name == "api.access"]
    assert lines, "expected one access-log line per request"
    msg = lines[-1].getMessage()
    assert "/api/search" in msg and "GET" in msg
