"""Tests for the Eval-set-builder admin routes (Task 30).

The /eval surface is admin-gated by the ``EVAL_ADMIN_TOKEN`` env var.
Without a token set, every route returns 403. With a token, the same
value must be supplied per request as a cookie or ``?token=...`` query
string. POST /api/eval/grade appends to the run JSONL; GET
/api/eval/metrics computes P@K + nDCG over the grades on file.
"""

from __future__ import annotations

import json

# ── Auth gating ───────────────────────────────────────────────────────────────


def test_eval_page_returns_403_without_admin_token_env(client, monkeypatch):
    monkeypatch.delenv("EVAL_ADMIN_TOKEN", raising=False)
    r = client.get("/eval")
    assert r.status_code == 403


def test_eval_page_returns_403_with_wrong_token(client, monkeypatch):
    monkeypatch.setenv("EVAL_ADMIN_TOKEN", "secret")
    r = client.get("/eval?token=not-the-secret")
    assert r.status_code == 403


def test_api_metrics_returns_403_without_token(client, monkeypatch):
    monkeypatch.delenv("EVAL_ADMIN_TOKEN", raising=False)
    r = client.get("/api/eval/metrics")
    assert r.status_code == 403


def test_api_grade_returns_403_without_token(client, monkeypatch):
    monkeypatch.delenv("EVAL_ADMIN_TOKEN", raising=False)
    r = client.post(
        "/api/eval/grade",
        data={"query_id": "q1", "scene_id": "jeca/1", "grade": "2"},
    )
    assert r.status_code == 403


# ── Persistence: POST /api/eval/grade ─────────────────────────────────────────


def test_post_grade_appends_to_run_jsonl(client, monkeypatch, tmp_path):
    """Grade persists to ``<eval_root>/<run_id>.jsonl`` as one JSON line."""

    monkeypatch.setenv("EVAL_ADMIN_TOKEN", "test-token")
    # Pin the eval root + run_id resolvers at the service layer so we
    # know exactly where the file lands.
    import api.services.eval_service as eval_service

    monkeypatch.setattr(eval_service, "_eval_root", lambda cfg: tmp_path)
    monkeypatch.setattr(eval_service, "_eval_run_id", lambda cfg: "default")

    r = client.post(
        "/api/eval/grade?token=test-token",
        data={"query_id": "q1", "scene_id": "jeca/1", "grade": "2"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True

    jsonl = tmp_path / "default.jsonl"
    assert jsonl.exists()
    lines = jsonl.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["query_id"] == "q1"
    assert record["scene_id"] == "jeca/1"
    assert record["grade"] == 2


def test_post_grade_accepts_skip(client, monkeypatch, tmp_path):
    monkeypatch.setenv("EVAL_ADMIN_TOKEN", "test-token")
    import api.services.eval_service as eval_service

    monkeypatch.setattr(eval_service, "_eval_root", lambda cfg: tmp_path)
    monkeypatch.setattr(eval_service, "_eval_run_id", lambda cfg: "default")

    r = client.post(
        "/api/eval/grade?token=test-token",
        data={"query_id": "q1", "scene_id": "jeca/1", "grade": "-1"},
    )
    assert r.status_code == 200
    record = json.loads((tmp_path / "default.jsonl").read_text().strip())
    assert record["grade"] == -1


# ── Metrics: GET /api/eval/metrics ────────────────────────────────────────────


def test_get_metrics_zeroed_for_empty_query(client, monkeypatch, tmp_path):
    monkeypatch.setenv("EVAL_ADMIN_TOKEN", "test-token")
    import api.services.eval_service as eval_service

    monkeypatch.setattr(eval_service, "_eval_root", lambda cfg: tmp_path)
    monkeypatch.setattr(eval_service, "_eval_run_id", lambda cfg: "default")

    r = client.get("/api/eval/metrics?token=test-token&query_id=q1")
    assert r.status_code == 200
    data = r.json()
    assert data["p_at_5"] == 0.0
    assert data["ndcg_at_5"] == 0.0
    assert data["inversions"] == 0


def test_get_metrics_reflects_grades(client, monkeypatch, tmp_path):
    """End-to-end: POST grades, then GET metrics returns non-zero P@K."""

    monkeypatch.setenv("EVAL_ADMIN_TOKEN", "test-token")
    import api.services.eval_service as eval_service

    monkeypatch.setattr(eval_service, "_eval_root", lambda cfg: tmp_path)
    monkeypatch.setattr(eval_service, "_eval_run_id", lambda cfg: "default")

    # Three grades on q1 → P@3 = 2/3
    for sid, grade in (("jeca/1", "3"), ("jeca/2", "2"), ("jeca/3", "0")):
        r = client.post(
            "/api/eval/grade?token=test-token",
            data={"query_id": "q1", "scene_id": sid, "grade": grade},
        )
        assert r.status_code == 200

    r = client.get("/api/eval/metrics?token=test-token&query_id=q1")
    assert r.status_code == 200
    data = r.json()
    # P@3 = 2/3 (HR + R count; IR does not)
    assert data["p_at_3"] == 2 / 3
    # nDCG should be in (0, 1] when at least one positive grade is present
    assert 0.0 < data["ndcg_at_5"] <= 1.0


def test_get_metrics_aggregate_lists_query_ids(client, monkeypatch, tmp_path):
    """Without ``query_id``, returns the set of graded query ids."""

    monkeypatch.setenv("EVAL_ADMIN_TOKEN", "test-token")
    import api.services.eval_service as eval_service

    monkeypatch.setattr(eval_service, "_eval_root", lambda cfg: tmp_path)
    monkeypatch.setattr(eval_service, "_eval_run_id", lambda cfg: "default")

    for qid, sid in (("q1", "jeca/1"), ("q2", "jeca/2"), ("q1", "jeca/3")):
        client.post(
            "/api/eval/grade?token=test-token",
            data={"query_id": qid, "scene_id": sid, "grade": "2"},
        )

    r = client.get("/api/eval/metrics?token=test-token")
    assert r.status_code == 200
    data = r.json()
    assert set(data["queries"]) == {"q1", "q2"}


# ── /eval page: token accepted, template tolerated ────────────────────────────


def test_eval_page_with_valid_token_is_authorized(client, monkeypatch, tmp_path):
    """A valid token clears the 403 gate. The Task-31 template may not
    exist yet, so 500 (TemplateNotFound) is acceptable here; what we
    check is that 403 is no longer returned."""

    monkeypatch.setenv("EVAL_ADMIN_TOKEN", "test-token")
    import api.services.eval_service as eval_service

    monkeypatch.setattr(eval_service, "_eval_root", lambda cfg: tmp_path)
    monkeypatch.setattr(eval_service, "_eval_run_id", lambda cfg: "default")

    r = client.get("/eval?token=test-token")
    # Task-30 ships the routes only; the eval/layout.html template lands
    # in Task 31. Either 200 (template present) or 500 (template
    # missing) is acceptable. 403 means auth gating mis-fired.
    assert r.status_code != 403
    assert r.status_code in {200, 500}


def test_eval_page_with_cookie_token_accepted(client, monkeypatch, tmp_path):
    monkeypatch.setenv("EVAL_ADMIN_TOKEN", "test-token")
    import api.services.eval_service as eval_service

    monkeypatch.setattr(eval_service, "_eval_root", lambda cfg: tmp_path)
    monkeypatch.setattr(eval_service, "_eval_run_id", lambda cfg: "default")

    client.cookies.set("eval_admin", "test-token")
    try:
        r = client.get("/eval")
        assert r.status_code != 403
    finally:
        client.cookies.delete("eval_admin")
