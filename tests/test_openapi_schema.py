"""A3: /docs is presentable — assert the OpenAPI schema carries typed models."""

from __future__ import annotations

import os


def test_openapi_lists_json_paths(client) -> None:
    spec = client.get("/openapi.json").json()
    paths = spec["paths"]
    for p in ["/api/eval/metrics", "/api/eval/grade", "/api/export/catalog.json", "/api/search"]:
        assert p in paths, f"{p} missing from OpenAPI"


def test_openapi_declares_response_models(client) -> None:
    spec = client.get("/openapi.json").json()
    schemas = spec["components"]["schemas"]
    assert "QueryMetrics" in schemas
    assert "GradeAck" in schemas
    # Typed search params surface as query parameters with enum/constraint metadata.
    search_get = spec["paths"]["/api/search"]["get"]
    param_names = {p["name"] for p in search_get.get("parameters", [])}
    assert {"q", "retriever", "modality", "top_k"} <= param_names


def test_docs_and_redoc_render(client) -> None:
    assert client.get("/docs").status_code == 200
    assert client.get("/redoc").status_code == 200
    assert "Cinemateca" in client.get("/docs").text


def test_metrics_response_validates_against_model(client, monkeypatch) -> None:
    """GET /api/eval/metrics with query_id=None returns {"queries": [...]}."""
    monkeypatch.setenv("EVAL_ADMIN_TOKEN", "t")
    r = client.get("/api/eval/metrics?token=t")
    assert r.status_code == 200
    body = r.json()
    # When query_id is omitted, compute_query_metrics returns {"queries": [...]}.
    assert "queries" in body


def test_metrics_per_query_keys(client, monkeypatch) -> None:
    """GET /api/eval/metrics?query_id=X returns the per-query metric keys."""
    monkeypatch.setenv("EVAL_ADMIN_TOKEN", "t")
    # A non-existent query_id returns zeroed-out metrics (the service guarantees this).
    r = client.get("/api/eval/metrics?token=t&query_id=does_not_exist")
    assert r.status_code == 200
    body = r.json()
    # Exact keys from compute_query_metrics (zeroed branch, no grades).
    assert "p_at_3" in body
    assert "p_at_5" in body
    assert "ndcg_at_5" in body
    assert "inversions" in body
    assert "histogram" in body


def test_grade_ack_response_shape(client, monkeypatch) -> None:
    """POST /api/eval/grade returns a GradeAck-shaped body."""
    monkeypatch.setenv("EVAL_ADMIN_TOKEN", "t")
    r = client.post(
        "/api/eval/grade?token=t",
        data={"query_id": "q1", "scene_id": "jeca/1", "grade": "2"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["query_id"] == "q1"
    assert body["scene_id"] == "jeca/1"
    assert body["grade"] == 2


def test_search_params_typed_in_openapi(client) -> None:
    """SearchParams constraints surface as typed query parameters in the OpenAPI spec.

    FastAPI inlines Depends(SearchParams) as individual query parameters — the
    model itself does not appear as a component schema, but each field's
    constraints (ge/le → minimum/maximum) are preserved in the inline schema.
    """
    spec = client.get("/openapi.json").json()
    params = {
        p["name"]: p
        for p in spec["paths"]["/api/search"]["get"].get("parameters", [])
    }
    # All typed params must be present
    for name in ("q", "retriever", "modality", "top_k", "reranker_enabled"):
        assert name in params, f"param {name!r} missing from /api/search"
    # top_k must carry ge/le constraints (minimum/maximum in JSON Schema)
    top_k_schema = params["top_k"]["schema"]
    assert top_k_schema.get("minimum") == 1, f"top_k minimum not set: {top_k_schema}"
    assert top_k_schema.get("maximum") == 200, f"top_k maximum not set: {top_k_schema}"
    # retriever and modality are string params (permissive — unknown values
    # fall back in the service layer, preserving backward compat)
    assert params["retriever"]["schema"]["type"] == "string"
    assert params["modality"]["schema"]["type"] == "string"
