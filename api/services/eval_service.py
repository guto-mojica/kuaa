"""Service layer for the Eval-set-builder routes (Tasks 30–31).

``_eval_root`` / ``_eval_run_id`` are intentionally module-level
functions so test fixtures can monkeypatch them to a tmp_path / fixed
run_id without constructing a full Config namespace. Admin gate
``require_admin`` moved here from api/routes/eval.py (A2 Task 5).
"""

from __future__ import annotations

import os
from typing import Any

from kuaa.eval.datasets import load_queries as _load_queries  # noqa: F401
from kuaa.eval.grader_metrics import (
    annotator_summary as _annotator_summary,
)
from kuaa.eval.grader_metrics import (
    build_iaa as _build_iaa,
)
from kuaa.eval.grader_metrics import (
    grader_initials as _grader_initials,  # noqa: F401
)
from kuaa.eval.grader_metrics import (
    grades_for_current_grader,
    histogram,
    inversions,
    ndcg_at_k,
    precision_at_k,
)
from kuaa.eval.grader_metrics import (
    initials as _initials,
)
from kuaa.eval.grader_metrics import (
    kappa_quality_label as _kappa_quality_label,  # noqa: F401
)
from kuaa.eval.grader_metrics import (
    other_grades_for_current as _other_grades_for_current,
)
from kuaa.eval.grader_metrics import (
    query_conflict_set as _query_conflict_set,
)
from kuaa.eval.grades import (
    EvalRun,
    Grade,
    load_run,
    load_run_per_annotator,
)
from kuaa.eval.grades import (
    first_ungraded as _first_ungraded,
)
from kuaa.eval.grades import (
    grades_by_query as _grades_by_query,  # noqa: F401
)
from kuaa.eval.grades import (
    grades_for_query as _grades_for_query,  # noqa: F401
)
from kuaa.eval.paths import (  # noqa: F401
    eval_root as _eval_root,
)
from kuaa.eval.paths import (
    eval_run_id as _eval_run_id,
)

# ── Public API ────────────────────────────────────────────────────────────────


def build_eval_context(cfg, *, request=None) -> dict[str, Any]:
    """Build the full /eval page context dict.

    All keys have safe zero/None defaults so the template renders on a
    fresh unseeded run. Cookie-driven identity is read from ``request``
    when provided; falls back to anon / off.
    """

    run_root = _eval_root(cfg)
    run_id = _eval_run_id(cfg)
    run = EvalRun(run_id=run_id, root=run_root)
    loaded = load_run(run)
    per_annotator = load_run_per_annotator(run)

    queries = _load_queries(run_root, run_id)

    # Resolve grader identity BEFORE current_query — _first_ungraded
    # needs grader_name, and the IAA + compare-mode helpers below also
    # need it to pick "the other annotator". Cookie-driven when a
    # request is provided; falls back to "anon" for direct callers (tests).
    grader_name = "anon"
    blind_mode = False
    compare_mode = False
    token = os.getenv("EVAL_ADMIN_TOKEN", "")
    if request is not None:
        grader_name = request.cookies.get("grader", "anon")
        blind_mode = request.cookies.get("eval_blind", "") == "1"
        compare_mode = request.cookies.get("eval_compare", "") == "1"
        token = request.cookies.get("eval_admin") or request.query_params.get("token") or token

    current_query = _first_ungraded(queries, per_annotator, grader_name) or (
        queries[0] if queries else None
    )

    annotator_count, iaa_kappa = _annotator_summary(per_annotator)
    grades_by_query = _grades_by_query(loaded)

    # Multi-annotator IAA bundle for the right-pane panel + the queue's
    # Conflict tab + the per-row compare-mode chip. All three views
    # consume the same source so the counts stay consistent.
    iaa = _build_iaa(per_annotator, current_grader=grader_name)
    query_conflict_set = _query_conflict_set(per_annotator)

    # UI fields (Task 31). All have safe zero defaults so the
    # template renders even on a fresh, unseeded run.
    graded_count = sum(1 for grades in grades_by_query.values() if grades)
    pending_count = max(0, len(queries) - graded_count)
    conflict_count = len(query_conflict_set)

    grades_for_current: dict[str, Grade] = {}
    grades_for_current_other: dict[str, Grade] = {}
    metrics: dict[str, Any] = {
        "p_at_3": 0.0,
        "p_at_5": 0.0,
        "ndcg_at_5": 0.0,
        "inversions": 0,
        "histogram": {},
    }
    result_count = 0
    if current_query is not None:
        cq_id = str(current_query.get("id", ""))
        if cq_id:
            # ``grades_for_current`` drives the .gb chip render in rows.html
            # — it must show THIS grader's grade (keyed from ``per_annotator``),
            # not the last-write-wins reduce of ``loaded.grades``. See
            # ``grades_for_current_grader`` for the fallback semantics when the
            # current grader has no per-annotator record yet.
            grades_for_current = grades_for_current_grader(
                per_annotator,
                loaded.grades,
                current_query_id=cq_id,
                grader_name=grader_name,
            )
            # Compare-mode counterpart — same query, other annotator.
            other_name = iaa.get("other", {}).get("name") if iaa.get("enabled") else None
            grades_for_current_other = _other_grades_for_current(
                per_annotator,
                current_query_id=cq_id,
                other_grader=other_name,
            )
            cq_grades = _grades_for_query(loaded, cq_id)
            if cq_grades:
                metrics = {
                    "p_at_3": precision_at_k(cq_grades, 3),
                    "p_at_5": precision_at_k(cq_grades, 5),
                    "ndcg_at_5": ndcg_at_k(cq_grades, 5),
                    "inversions": inversions(cq_grades),
                    "histogram": histogram(cq_grades),
                }
        results = current_query.get("results")
        if isinstance(results, list):
            result_count = len(results)

    return {
        # Data layer
        "run_id": run_id,
        "queries": queries,
        "current_query": current_query,
        "grades_by_query": grades_by_query,
        "annotator_count": annotator_count,
        "iaa_kappa": iaa_kappa,
        # UI layer
        "graded_count": graded_count,
        "pending_count": pending_count,
        "conflict_count": conflict_count,
        "query_conflict_set": query_conflict_set,
        "iaa": iaa,
        "grades_for_current_other": grades_for_current_other,
        "metrics": metrics,
        "grades_for_current": grades_for_current,
        "grader_name": grader_name,
        "grader_initials": _initials(grader_name),
        "token": token,
        "blind_mode": blind_mode,
        "compare_mode": compare_mode,
        "current_row_scene_id": None,
        "result_count": result_count,
        "session_elapsed": "00:00",
    }


def compute_query_metrics(cfg, *, query_id: str | None = None) -> dict[str, Any]:
    """Return P@K/nDCG/inversions/histogram for ``query_id`` (zeroed when no grades).

    When ``query_id`` is None, returns the graded query id list instead.
    """
    run_root = _eval_root(cfg)
    run = EvalRun(run_id=_eval_run_id(cfg), root=run_root)
    loaded = load_run(run)

    if query_id is None:
        return {"queries": sorted({qid for (qid, _) in loaded.grades.keys()})}

    grades_list = _grades_for_query(loaded, query_id)
    if not grades_list:
        return {
            "p_at_3": 0.0,
            "p_at_5": 0.0,
            "ndcg_at_5": 0.0,
            "inversions": 0,
            "histogram": {},
        }

    return {
        "p_at_3": precision_at_k(grades_list, 3),
        "p_at_5": precision_at_k(grades_list, 5),
        "ndcg_at_5": ndcg_at_k(grades_list, 5),
        "inversions": inversions(grades_list),
        "histogram": histogram(grades_list),
    }


# ``_underscored`` helpers above are re-exported from kuaa.eval.* at the top.
# The ``as _underscored`` aliases preserve names that test fixtures monkeypatch.


def require_admin(request) -> None:
    """Raise HTTPException(403) unless the request bears a valid EVAL_ADMIN_TOKEN.

    Moved from api/routes/eval.py (A2 Task 5). Acceptable to raise HTTP-shaped
    exceptions here: the gate is tiny, reused, and A4 leaves it as-is.
    """
    from fastapi import HTTPException, status

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
