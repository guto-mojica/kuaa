"""Eval set builder routes — admin-gated by ``EVAL_ADMIN_TOKEN`` env var.

Admin gate and context builders live in :mod:`api.services.eval_service`
(A2 Task 5).
"""

from __future__ import annotations

from fastapi import APIRouter, Form, HTTPException, Request, status

import api.services.eval_service as _eval_svc
from api.deps import get_config, make_ctx
from api.schemas import GradeAck, QueryMetrics
from api.services.eval_service import (
    build_eval_context,
    compute_query_metrics,
    require_admin,
)
from api.templates import templates
from kuaa.eval.grades import EvalRun, Grade, save_grade

router = APIRouter()


@router.get("/eval")
def eval_page(request: Request):
    """Render the eval grading UI (403 when EVAL_ADMIN_TOKEN is unset)."""
    require_admin(request)
    cfg = get_config()
    ctx = build_eval_context(cfg, request=request)
    return templates.TemplateResponse(
        request,
        "eval/layout.html",
        make_ctx(request, **ctx),
    )


@router.post("/api/eval/grade", response_model=GradeAck)
def post_grade(
    request: Request,
    query_id: str = Form(...),
    scene_id: str = Form(...),
    grade: int = Form(...),
) -> GradeAck:
    """Append one grade to the active run JSONL."""
    require_admin(request)
    try:
        grade_enum = Grade(grade)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid grade value: {grade}",
        ) from exc

    cfg = get_config()
    run_root = _eval_svc._eval_root(cfg)
    run_id = _eval_svc._eval_run_id(cfg)
    grader = request.cookies.get("grader", "anon")

    run = EvalRun(run_id=run_id, root=run_root)
    save_grade(
        run,
        query_id=query_id,
        scene_id=scene_id,
        grader=grader,
        grade=grade_enum,
    )
    return GradeAck(ok=True, query_id=query_id, scene_id=scene_id, grade=int(grade_enum))


@router.get("/api/eval/metrics", response_model=QueryMetrics)
def get_metrics(request: Request, query_id: str | None = None) -> QueryMetrics:
    """Return P@K, nDCG, inversions and histogram for one query.

    When ``query_id`` is omitted, returns the list of graded query IDs
    (``{"queries": [...]}``) rather than per-query metrics.
    """
    require_admin(request)
    cfg = get_config()
    raw = compute_query_metrics(cfg, query_id=query_id)
    return QueryMetrics(**raw)
