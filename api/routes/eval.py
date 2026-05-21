"""Eval set builder routes — admin-gated by ``EVAL_ADMIN_TOKEN`` env var.

Three endpoints land in Task 30:

* ``GET /eval`` — the grading UI (Task 31 supplies the template). 403
  when ``EVAL_ADMIN_TOKEN`` is unset or the request token mismatches.
* ``POST /api/eval/grade`` — append a grade to the run JSONL.
* ``GET /api/eval/metrics`` — read live metrics (P@K / nDCG /
  inversions / histogram) for one query, or the list of graded
  query ids when no ``query_id`` is given.

Auth model (deliberately minimal)
---------------------------------
A single shared token in the ``EVAL_ADMIN_TOKEN`` env var. Clients
present it either as a cookie (``eval_admin``) or a ``?token=...``
query string. When the env var is empty/unset the entire surface
returns 403 — the eval builder is "off" until the operator opts in.
This is intentionally lighter than the project's planned user model:
the eval set is curator-only, the deployment is internal, and we
don't want the launch to depend on auth scaffolding that isn't
needed elsewhere yet.
"""

from __future__ import annotations

import os

from fastapi import APIRouter, Form, HTTPException, Request, status

from api.deps import get_config, make_ctx
from api.services.eval_service import build_eval_context, compute_query_metrics
from api.templates import templates
from cinemateca.eval.grades import EvalRun, Grade, save_grade

router = APIRouter()


# ── Admin gate ────────────────────────────────────────────────────────────────


def _check_admin(request: Request) -> None:
    """Raise 403 unless the request bears a valid admin token.

    The expected value comes from the ``EVAL_ADMIN_TOKEN`` env var.
    An empty/unset env var disables the entire eval surface — every
    route then returns 403 with a "disabled" message so we don't leak
    behaviour to anonymous clients.
    """

    expected = os.getenv("EVAL_ADMIN_TOKEN", "")
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Eval set builder is disabled. Set EVAL_ADMIN_TOKEN to enable.",
        )
    token = request.cookies.get("eval_admin") or request.query_params.get("token") or ""
    if token != expected:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Eval set builder requires a valid admin token.",
        )


# ── /eval (page) ──────────────────────────────────────────────────────────────


@router.get("/eval")
def eval_page(request: Request):
    """Render the eval grading UI.

    Task 30 ships the route + context builder; the
    ``eval/layout.html`` template lands in Task 31. If the template
    is missing the response is still authorised — Jinja2's
    ``TemplateNotFound`` simply propagates as a 500. This is
    intentional: it tells the operator the backend is wired but the
    UI hasn't shipped yet, instead of silently returning 200 with a
    blank page.
    """

    _check_admin(request)
    cfg = get_config()
    ctx = build_eval_context(cfg, request=request)
    return templates.TemplateResponse(
        request,
        "eval/layout.html",
        make_ctx(request, **ctx),
    )


# ── /api/eval/grade ───────────────────────────────────────────────────────────


@router.post("/api/eval/grade")
def post_grade(
    request: Request,
    query_id: str = Form(...),
    scene_id: str = Form(...),
    grade: int = Form(...),
):
    """Append one grade to the active run JSONL.

    The grade value is validated against the ``Grade`` enum
    (``-1, 0, 1, 2, 3``); anything else returns 422. The grader id
    is read from the ``grader`` cookie when present, falling back to
    ``"anon"``. The route returns a tiny JSON ack — the UI doesn't
    re-render off this response; it issues a follow-up HTMX request
    for the next row instead (Task 32).
    """

    _check_admin(request)
    try:
        grade_enum = Grade(grade)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid grade value: {grade}",
        ) from exc

    # Resolve eval root + run id via the service-layer helpers so test
    # ``monkeypatch.setattr`` on those helpers takes effect here too.
    from api.services import eval_service

    cfg = get_config()
    run_root = eval_service._eval_root(cfg)
    run_id = eval_service._eval_run_id(cfg)
    grader = request.cookies.get("grader", "anon")

    run = EvalRun(run_id=run_id, root=run_root)
    save_grade(
        run,
        query_id=query_id,
        scene_id=scene_id,
        grader=grader,
        grade=grade_enum,
    )
    return {"ok": True, "query_id": query_id, "scene_id": scene_id, "grade": int(grade_enum)}


# ── /api/eval/metrics ─────────────────────────────────────────────────────────


@router.get("/api/eval/metrics")
def get_metrics(request: Request, query_id: str | None = None):
    """Return P@K, nDCG, inversions and histogram for one query.

    Omit ``query_id`` to get the list of graded query ids (used by
    the queue panel). The metrics dict is always returned with all
    keys present (zeroed when no grades exist yet) so the template
    can read them unconditionally.
    """

    _check_admin(request)
    cfg = get_config()
    return compute_query_metrics(cfg, query_id=query_id)
