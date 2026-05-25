"""Service layer for the Eval-set-builder routes (Tasks 30–31).

Builds the /eval page context, computes per-query metrics for the
``/api/eval/metrics`` endpoint, and resolves the configured eval root
+ run id (with safe defaults). Persistence lives in
``cinemateca.eval.grades``; metrics math lives in
``cinemateca.eval.grader_metrics``.

Task 31 (the standalone grading UI) consumes ``build_eval_context``;
the Task-30 routes consume ``compute_query_metrics``. The two
``_eval_root`` / ``_eval_run_id`` resolvers are intentionally
module-level functions (not config-namespace lookups inlined into the
routes) so the test fixtures can ``monkeypatch.setattr`` them to a
tmp_path / fixed run_id without having to construct a full Config
namespace.
"""

from __future__ import annotations

import os
from typing import Any

from cinemateca.eval.datasets import load_queries as _load_queries  # noqa: F401
from cinemateca.eval.grader_metrics import (
    annotator_summary as _annotator_summary,
)
from cinemateca.eval.grader_metrics import (
    build_iaa as _build_iaa,
)
from cinemateca.eval.grader_metrics import (
    grader_initials as _grader_initials,  # noqa: F401
)
from cinemateca.eval.grader_metrics import (
    histogram,
    inversions,
    ndcg_at_k,
    precision_at_k,
)
from cinemateca.eval.grader_metrics import (
    initials as _initials,
)
from cinemateca.eval.grader_metrics import (
    kappa_quality_label as _kappa_quality_label,  # noqa: F401
)
from cinemateca.eval.grader_metrics import (
    other_grades_for_current as _other_grades_for_current,
)
from cinemateca.eval.grader_metrics import (
    query_conflict_set as _query_conflict_set,
)
from cinemateca.eval.grades import (
    EvalRun,
    Grade,
    load_run,
    load_run_per_annotator,
)
from cinemateca.eval.grades import (
    grades_by_query as _grades_by_query,  # noqa: F401
)
from cinemateca.eval.grades import (
    grades_for_query as _grades_for_query,  # noqa: F401
)
from cinemateca.eval.paths import (  # noqa: F401
    eval_root as _eval_root,
)
from cinemateca.eval.paths import (
    eval_run_id as _eval_run_id,
)

# ── Public API ────────────────────────────────────────────────────────────────


def build_eval_context(cfg, *, request=None) -> dict[str, Any]:
    """Build the full /eval page context dict.

    Keys returned:
      run_id, queries, current_query, grades_by_query, annotator_count,
      iaa_kappa, graded_count, pending_count, conflict_count,
      query_conflict_set, iaa, grades_for_current_other, metrics,
      grades_for_current, grader_name, grader_initials, token,
      blind_mode, compare_mode, current_row_scene_id, result_count,
      session_elapsed.

    All keys have safe zero/None defaults so the template renders on a
    fresh unseeded run. Cookie-driven identity (grader, blind, compare)
    is read from ``request`` when provided; falls back to anon / off.
    """

    run_root = _eval_root(cfg)
    run_id = _eval_run_id(cfg)
    run = EvalRun(run_id=run_id, root=run_root)
    loaded = load_run(run)
    per_annotator = load_run_per_annotator(run)

    queries = _load_queries(run_root, run_id)
    current_query = queries[0] if queries else None

    # Resolve grader identity FIRST — IAA + compare-mode helpers below
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
            # ``grades_for_current`` drives the .gb chip render in
            # rows.html — it must show THIS grader's grade, not the
            # last-write-wins reduce of ``loaded.grades`` (which would
            # surface the other annotator's grade in a multi-grader
            # run and make every disagree-by-≥2 chip vanish). Read
            # from ``per_annotator`` keyed by the current grader,
            # falling back to the collapsed reduce when no
            # per-annotator entry exists yet (e.g. anonymous viewers).
            for (qid, scene_id), by_who in per_annotator.items():
                if qid != cq_id:
                    continue
                if grader_name in by_who:
                    grades_for_current[str(scene_id)] = by_who[grader_name].grade
                elif by_who:
                    # No record from current grader — surface whatever
                    # is on file so the row isn't ungraded-looking.
                    # Prefer the collapsed-loaded.grades entry to
                    # preserve the prior single-grader semantics.
                    canonical = loaded.grades.get((qid, scene_id))
                    if canonical is not None:
                        grades_for_current[str(scene_id)] = canonical.grade
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
    """Compute P@K + nDCG + inversions + histogram for one query.

    When ``query_id`` is None, returns the list of graded query ids on
    the run (``{"queries": [...]}``) — used by the /eval page to drive
    the queue. When a ``query_id`` is given, returns the metric bundle
    even if no grades exist yet (zeroed values), so the right-pane
    template never has to special-case "first grade not yet saved".
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

# All ``_underscored`` helpers are re-exported from cinemateca.eval.* at the top.
# The ``as _underscored`` aliases preserve names that test fixtures monkeypatch.
