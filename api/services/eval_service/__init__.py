"""Service layer for the Eval-set-builder routes.

``_eval_root`` / ``_eval_run_id`` are intentionally module-level
functions so test fixtures can monkeypatch them to a tmp_path / fixed
run_id without constructing a full Config namespace.

A package rather than a module, for the same reason
``api/services/scenes/`` is one: the file outgrew the 250-line
``api/services/**`` budget. What stayed here is what the patchable seam
reaches — every caller of ``_eval_root`` / ``_eval_run_id`` resolves them
through this module's globals, so moving one out would put it beyond the
reach of a fixture that patches this name. The two private submodules hold
the parts that never call them.
"""

from __future__ import annotations

import os
from typing import Any

from api.services.eval_service._admin import require_admin
from api.services.eval_service._current_query import build_current_query_view
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
    other_grades_for_current as _other_grades_for_current,  # noqa: F401
)
from kuaa.eval.grader_metrics import (
    query_conflict_set as _query_conflict_set,
)
from kuaa.eval.grades import (
    EvalRun,
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

__all__ = ["build_eval_context", "compute_query_metrics", "require_admin"]

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
    # Blind by default for a pooled run: its rows already arrive shuffled with
    # score blanked, and the point of building it that way is that the grader
    # cannot read the retriever's opinion off the page. An explicit cookie
    # still wins in both directions — this only moves the default.
    pooled_run = any(q.get("pool_variants") for q in queries)
    grader_name = "anon"
    blind_mode = pooled_run
    compare_mode = False
    token = os.getenv("EVAL_ADMIN_TOKEN", "")
    if request is not None:
        grader_name = request.cookies.get("grader", "anon")
        blind_cookie = request.cookies.get("eval_blind", "")
        blind_mode = blind_cookie == "1" if blind_cookie else pooled_run
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

    # Hydrates current_query["results"] in place — see _current_query.
    cq_view = build_current_query_view(
        current_query,
        cfg=cfg,
        loaded=loaded,
        per_annotator=per_annotator,
        iaa=iaa,
        grader_name=grader_name,
    )

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
        "grades_for_current_other": cq_view["grades_for_current_other"],
        "metrics": cq_view["metrics"],
        "grades_for_current": cq_view["grades_for_current"],
        "grader_name": grader_name,
        "grader_initials": _initials(grader_name),
        "token": token,
        "blind_mode": blind_mode,
        "compare_mode": compare_mode,
        "current_row_scene_id": None,
        "result_count": cq_view["result_count"],
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
