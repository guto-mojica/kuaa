"""A8: every response carries X-Request-ID; the id is stable per request."""

from __future__ import annotations


def test_response_has_request_id(client) -> None:
    r = client.get("/health")
    assert "X-Request-ID" in r.headers
    assert r.headers["X-Request-ID"]


def test_request_id_echoed_when_supplied(client) -> None:
    # If the client supplies an id, the middleware should echo it (F5 contract).
    r = client.get("/health", headers={"X-Request-ID": "abc-123"})
    assert r.headers["X-Request-ID"] == "abc-123"


def test_sse_stream_carries_request_id(client) -> None:
    with client.stream("GET", "/api/pipeline/stream/deadbeef") as resp:
        assert "X-Request-ID" in resp.headers
