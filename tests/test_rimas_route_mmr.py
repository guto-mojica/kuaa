"""/tab/rimas, /api/rimas/echoes, /api/rimas/inspector accept lambda + k_candidates."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def _patch_capture(monkeypatch):
    """Monkeypatch the build_rimas_context import in api.routes.rimas to capture kwargs."""
    from api.routes import rimas as rimas_route

    captured: dict = {}

    def fake(*args, **kwargs):
        captured.update(kwargs)
        return {
            "anchor_film": None,
            "anchor_scene": None,
            "echoes": [],
            "selected_echo": None,
            "selected_echo_id": None,
            "shared_tags": [],
            "k": 8,
            "mmr_lambda": 0.5,
            "k_candidates": 30,
            "threshold": 0.75,
            "library_has_scenes": False,
        }

    monkeypatch.setattr(rimas_route, "build_rimas_context", fake)
    return captured


@pytest.mark.parametrize("path", ["/tab/rimas", "/api/rimas/echoes", "/api/rimas/inspector"])
def test_route_forwards_lambda_query_to_service(path, monkeypatch):
    from api.server import app

    captured = _patch_capture(monkeypatch)
    client = TestClient(app)
    r = client.get(path, params={"anchor": "any/1", "lambda": "0.3", "k_candidates": "25"})
    assert r.status_code == 200, r.text
    assert captured.get("lambda_diversity") == 0.3
    assert captured.get("k_candidates") == 25


@pytest.mark.parametrize("path", ["/tab/rimas", "/api/rimas/echoes", "/api/rimas/inspector"])
def test_route_defaults_to_none_when_lambda_absent(path, monkeypatch):
    """Without ?lambda=, service receives None -> resolves to cfg or hard default."""
    from api.server import app

    captured = _patch_capture(monkeypatch)
    client = TestClient(app)
    r = client.get(path, params={"anchor": "any/1"})
    assert r.status_code == 200
    assert captured.get("lambda_diversity") is None
    assert captured.get("k_candidates") is None
