"""Tests for eval session resume — build_eval_context picks the first ungraded query.

Deliverable: E5 (agent part) — session resume.

Two cases:
  1. A grader who has graded ALL scenes of query id 1 resumes to query id 2.
  2. A grader with NO grades (fresh) gets query id 1 (back-compat).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from cinemateca.eval.grades import EvalRun, Grade, save_grade
from cinemateca.eval.seed import SAMPLE_QUERIES, write_seed

# ── helpers ─────────────────────────────────────────────────────────────────


def _fake_request(grader: str) -> SimpleNamespace:
    """Minimal request object: cookies dict + empty query_params."""

    class _Cookies(dict):
        def get(self, key, default=""):  # type: ignore[override]
            return super().get(key, default)

    class _QueryParams(dict):
        def get(self, key, default=""):  # type: ignore[override]
            return super().get(key, default)

    req = SimpleNamespace()
    req.cookies = _Cookies({"grader": grader})
    req.query_params = _QueryParams()
    return req


def _make_cfg(run_root: Path, run_id: str) -> SimpleNamespace:
    """Minimal config that eval_root/eval_run_id resolve correctly."""
    eval_ns = SimpleNamespace(root=str(run_root), run_id=run_id)
    return SimpleNamespace(eval=eval_ns)


# ── tests ────────────────────────────────────────────────────────────────────


def test_resume_picks_first_ungraded_for_grader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A grader who completed query 1 resumes to query 2; a fresh grader starts at 1."""

    run_id = "r"
    write_seed(tmp_path, run_id, count=5)
    run = EvalRun(run_id=run_id, root=tmp_path)

    # Grade ALL scenes of query id 1 (id==1 → str "1") for grader "rg".
    # The seed's query 1 has 9 results; scene_ids are the integer values
    # stored in SAMPLE_QUERIES[0]["results"][*]["scene_id"].
    q1_scene_ids = [str(r["scene_id"]) for r in SAMPLE_QUERIES[0]["results"]]
    for sid in q1_scene_ids:
        save_grade(run, query_id="1", scene_id=sid, grader="rg", grade=Grade.RELEVANT)

    # Pin the service-layer resolvers to our tmp run.
    import api.services.eval_service as eval_service

    monkeypatch.setattr(eval_service, "_eval_root", lambda cfg: tmp_path)
    monkeypatch.setattr(eval_service, "_eval_run_id", lambda cfg: run_id)

    cfg = _make_cfg(tmp_path, run_id)

    # Grader "rg" has graded query 1 — should resume to query 2.
    ctx_rg = eval_service.build_eval_context(cfg, request=_fake_request("rg"))
    assert ctx_rg["current_query"] is not None
    assert (
        str(ctx_rg["current_query"]["id"]) == "2"
    ), f"expected resume to query 2, got {ctx_rg['current_query']['id']!r}"

    # Grader "fresh" has no grades — should start at query 1 (back-compat).
    ctx_fresh = eval_service.build_eval_context(cfg, request=_fake_request("fresh"))
    assert ctx_fresh["current_query"] is not None
    assert (
        str(ctx_fresh["current_query"]["id"]) == "1"
    ), f"expected fresh grader at query 1, got {ctx_fresh['current_query']['id']!r}"


def test_resume_all_graded_falls_back_to_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A grader who has graded every query falls back to query 1 (not None)."""

    run_id = "r"
    write_seed(tmp_path, run_id, count=2)
    run = EvalRun(run_id=run_id, root=tmp_path)

    # Grade at least one scene for both queries as "done".
    for qid in ("1", "2"):
        save_grade(run, query_id=qid, scene_id="99", grader="rg", grade=Grade.RELEVANT)

    import api.services.eval_service as eval_service

    monkeypatch.setattr(eval_service, "_eval_root", lambda cfg: tmp_path)
    monkeypatch.setattr(eval_service, "_eval_run_id", lambda cfg: run_id)

    cfg = _make_cfg(tmp_path, run_id)
    ctx = eval_service.build_eval_context(cfg, request=_fake_request("rg"))

    # No ungraded query → fall back to query 1 rather than None.
    assert ctx["current_query"] is not None
    assert str(ctx["current_query"]["id"]) == "1"
