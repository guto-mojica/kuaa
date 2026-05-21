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


# ── Task 31: standalone 3-pane layout ─────────────────────────────────────────


def test_eval_layout_renders_3pane_grid(client, monkeypatch, tmp_path):
    """The Task-31 standalone shell ships the .ev-app grid + admin
    strip + header + 3-pane body + footer. None of the Mojica chrome
    classes leak in (no TopBar / IconRail / LeftPane on the eval
    surface — it's full-viewport, admin-only)."""

    monkeypatch.setenv("EVAL_ADMIN_TOKEN", "test-token")
    import api.services.eval_service as eval_service

    monkeypatch.setattr(eval_service, "_eval_root", lambda cfg: tmp_path)
    monkeypatch.setattr(eval_service, "_eval_run_id", lambda cfg: "default")

    r = client.get("/eval?token=test-token")
    assert r.status_code == 200, r.text
    html = r.text
    # Root + four grid rows (admin / top / body / bot).
    assert 'class="ev-app"' in html
    assert 'class="ev-admin"' in html
    assert 'class="ev-top"' in html
    assert 'class="ev-body"' in html
    assert 'class="ev-bot"' in html
    # Stylesheet is loaded on this page only.
    assert "eval.css" in html
    # Standalone shell — no Mojica chrome containers.
    assert "ch-body" not in html
    assert "ch-rail" not in html


def test_eval_queue_pane_present(client, monkeypatch, tmp_path):
    """The left pane ships filter tabs + toggles even when the run
    has no queries yet."""

    monkeypatch.setenv("EVAL_ADMIN_TOKEN", "test-token")
    import api.services.eval_service as eval_service

    monkeypatch.setattr(eval_service, "_eval_root", lambda cfg: tmp_path)
    monkeypatch.setattr(eval_service, "_eval_run_id", lambda cfg: "default")

    r = client.get("/eval?token=test-token")
    assert r.status_code == 200
    html = r.text
    # Left-pane container + filter-tab buttons.
    assert 'class="ev-lp"' in html
    assert 'data-filter="todas"' in html
    assert 'data-filter="pendentes"' in html
    assert 'data-filter="conflito"' in html
    # Toggles for blind/compare modes.
    assert 'data-toggle="blind"' in html
    assert 'data-toggle="compare"' in html


def test_eval_metrics_pane_zero_state(client, monkeypatch, tmp_path):
    """The right pane renders all four big-metric cards on an empty
    run with zeroed values (the template must not crash when no
    grades / no current_query exist)."""

    monkeypatch.setenv("EVAL_ADMIN_TOKEN", "test-token")
    import api.services.eval_service as eval_service

    monkeypatch.setattr(eval_service, "_eval_root", lambda cfg: tmp_path)
    monkeypatch.setattr(eval_service, "_eval_run_id", lambda cfg: "default")

    r = client.get("/eval?token=test-token")
    assert r.status_code == 200
    html = r.text
    # Right pane container + metric labels (we don't assert exact
    # values because future locale changes could move them, but the
    # labels themselves are part of the stable layout contract).
    assert 'class="ev-rp"' in html
    assert "PRECISION@5" in html
    assert "NDCG@5" in html
    assert "PRECISION@3" in html
    assert "INVERSIONS" in html
    # Save / skip action buttons.
    assert 'data-action="save-advance"' in html
    assert 'data-action="skip"' in html


# ── Task 32: keyboard router asset wiring ─────────────────────────────────────


def test_eval_js_asset_served(client):
    """The eval.js router asset is served from /static/js/eval.js and
    contains the key entry-point symbols. We assert on stable code
    landmarks (`.ev-app` selector, `gradeRow` function) rather than
    line counts so future polish edits don't break the test."""

    r = client.get("/static/js/eval.js")
    assert r.status_code == 200
    body = r.text
    assert ".ev-app" in body  # selector that bootstraps the script
    assert "gradeRow" in body  # POST helper that wraps /api/eval/grade
    # Keyboard contract — the four keys named in the footer legend.
    assert "ArrowDown" in body or "'j'" in body
    assert "/api/eval/grade" in body


def test_eval_layout_loads_eval_js(client, monkeypatch, tmp_path):
    """The standalone eval shell pulls in /static/js/eval.js. We also
    confirm the script tag carries `defer` so it doesn't block the
    initial paint and runs after the .ev-app element exists in the
    DOM (the script's bootstrap depends on that)."""

    monkeypatch.setenv("EVAL_ADMIN_TOKEN", "test-token")
    import api.services.eval_service as eval_service

    monkeypatch.setattr(eval_service, "_eval_root", lambda cfg: tmp_path)
    monkeypatch.setattr(eval_service, "_eval_run_id", lambda cfg: "default")

    r = client.get("/eval?token=test-token")
    assert r.status_code == 200
    html = r.text
    # The reference is rendered through url_for, so the host shows up too,
    # but the path component is stable.
    assert "js/eval.js" in html
    # The eval.js tag must be defer-loaded (mojica.js is, eval.js follows).
    assert 'js/eval.js" defer' in html or "js/eval.js' defer" in html


def test_eval_grade_then_metrics_round_trip(client, monkeypatch, tmp_path):
    """End-to-end backend round trip the JS router will issue: POST a
    grade, then GET metrics returns a non-zero P@K. This is the
    contract the keyboard router depends on — the JS side itself is
    DOM-only and can't be exercised in pytest, but the network shape
    must hold."""

    monkeypatch.setenv("EVAL_ADMIN_TOKEN", "test-token")
    import api.services.eval_service as eval_service

    monkeypatch.setattr(eval_service, "_eval_root", lambda cfg: tmp_path)
    monkeypatch.setattr(eval_service, "_eval_run_id", lambda cfg: "default")

    # Same payload shape as gradeRow() in eval.js: form-encoded POST.
    r1 = client.post(
        "/api/eval/grade?token=test-token",
        data={"query_id": "q1", "scene_id": "jeca/1", "grade": "2"},
    )
    assert r1.status_code == 200, r1.text
    assert r1.json()["ok"] is True

    # Same GET shape as refreshMetrics() in eval.js.
    r2 = client.get("/api/eval/metrics?query_id=q1&token=test-token")
    assert r2.status_code == 200
    data = r2.json()
    # One RELEVANT grade (=2) over a single-item slate yields P@K = 1.0.
    assert data["p_at_5"] == 1.0
    assert data["p_at_3"] == 1.0
    # And all four keys the JS expects are present on every response.
    for key in ("p_at_5", "p_at_3", "ndcg_at_5", "inversions"):
        assert key in data


# ── Task 33: seeded queries appear in the queue ───────────────────────────────


def test_eval_page_shows_seeded_queries(client, monkeypatch, tmp_path):
    """After running ``cinemateca eval seed``, the /eval page renders the
    seeded queries in the left-pane queue. We assert on the row count by
    counting the ``ev-q`` queue-row class (one per query — current and
    pending entries both carry it)."""

    monkeypatch.setenv("EVAL_ADMIN_TOKEN", "test-token")
    import api.services.eval_service as eval_service

    monkeypatch.setattr(eval_service, "_eval_root", lambda cfg: tmp_path)
    monkeypatch.setattr(eval_service, "_eval_run_id", lambda cfg: "default")

    from cinemateca.eval.seed import write_seed

    write_seed(tmp_path, "default", count=3)

    r = client.get("/eval?token=test-token")
    assert r.status_code == 200, r.text
    html = r.text
    # Three queue rows in the left pane. The template emits
    # ``class="ev-q"`` for pending rows and ``class="ev-q cur"`` for the
    # one that's currently selected; counting both substrings covers
    # both. There's no other ``ev-q`` token in the page chrome so this
    # count is exact.
    queue_row_hits = html.count('class="ev-q"') + html.count('class="ev-q cur"')
    assert (
        queue_row_hits == 3
    ), f"expected 3 queue rows in /eval after seeding, got {queue_row_hits}"
    # The first seeded query's text should appear in the queue pane.
    assert "duas pessoas conversando" in html
