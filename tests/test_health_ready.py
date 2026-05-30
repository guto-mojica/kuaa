"""A5: liveness + readiness endpoints."""

from __future__ import annotations

from pathlib import Path


def test_health_is_ok(client) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ready_ok_on_valid_config(client) -> None:
    r = client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is True
    assert body["checks"]["data_dir_readable"] is True
    assert body["checks"]["registry_parseable"] is True


def test_ready_fails_when_data_dir_missing(client, tmp_config, monkeypatch) -> None:
    # Point the data dir at a non-existent path and assert /ready degrades to 503.
    missing = Path(tmp_config.paths.data_dir) / "nope-removed"
    monkeypatch.setattr(tmp_config.paths, "data_dir", missing)
    r = client.get("/ready")
    assert r.status_code == 503
    assert r.json()["ready"] is False
