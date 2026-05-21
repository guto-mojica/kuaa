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

import json
import os
from pathlib import Path
from typing import Any

from cinemateca.eval.grader_metrics import (
    cohen_kappa,
    histogram,
    inversions,
    ndcg_at_k,
    precision_at_k,
)
from cinemateca.eval.grades import EvalRun, Grade, LoadedRun, load_run

# ── Public API ────────────────────────────────────────────────────────────────


def build_eval_context(cfg, *, request=None) -> dict[str, Any]:
    """Build the full /eval page context.

    Task 30 ships the data-layer keys; Task 31 adds the UI keys that
    the standalone /eval template (web/templates/eval/layout.html and
    its queue/rows/metrics partials) reads. The contract:

    Data layer (Task 30):
      * ``run_id`` — current run identifier (string)
      * ``queries`` — list of query dicts loaded from
        ``<root>/<run_id>.queries.json`` (empty when not seeded)
      * ``current_query`` — the first query in the queue (None when
        empty)
      * ``grades_by_query`` — ``query_id -> list[Grade]`` for the
        right pane's "graded so far" indicators
      * ``annotator_count`` — distinct ``grader`` ids seen on the
        JSONL
      * ``iaa_kappa`` — Cohen's κ between the two most-active graders
        when ``annotator_count >= 2`` else 0.0

    UI layer (Task 31):
      * ``graded_count`` — number of queries with at least one grade
      * ``pending_count`` — number of queries with no grades yet
      * ``conflict_count`` — placeholder (Task 33 wires multi-annotator
        conflict detection); 0 until then
      * ``metrics`` — metric bundle for ``current_query`` (zeroed
        when no grades exist) so the right pane never crashes on
        first paint
      * ``grades_for_current`` — ``scene_id -> Grade`` for the
        current query (the rows template colours its grade buttons
        from this dict)
      * ``grader_name`` / ``grader_initials`` — read from the
        ``grader`` cookie when the request is provided; defaults to
        "anon" / "AN"
      * ``token`` — the admin token (echoed back into query links so
        the queue's per-query anchors stay authorised)
      * ``blind_mode`` / ``compare_mode`` — UI toggles (read from
        ``blind`` / ``compare`` cookies); both default off
      * ``current_row_scene_id`` — None until Task 32's keyboard
        router moves the row cursor
      * ``result_count`` — number of candidate results on the
        current query (or 0)
      * ``session_elapsed`` — placeholder mm:ss string; Task 33
        wires real timing
    """

    run_root = _eval_root(cfg)
    run_id = _eval_run_id(cfg)
    run = EvalRun(run_id=run_id, root=run_root)
    loaded = load_run(run)

    queries = _load_queries(run_root, run_id)
    current_query = queries[0] if queries else None

    annotator_count, iaa_kappa = _annotator_summary(loaded)
    grades_by_query = _grades_by_query(loaded)

    # UI fields (Task 31). All have safe zero defaults so the
    # template renders even on a fresh, unseeded run.
    graded_count = sum(1 for grades in grades_by_query.values() if grades)
    pending_count = max(0, len(queries) - graded_count)
    conflict_count = 0  # Task 33 wires real multi-annotator conflict detection.

    grades_for_current: dict[str, Grade] = {}
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
            for (qid, scene_id), entry in loaded.grades.items():
                if qid == cq_id:
                    grades_for_current[str(scene_id)] = entry.grade
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

    # Grader identity is cookie-driven on the route. When no request
    # is passed (e.g. tests calling build_eval_context directly) we
    # fall back to the anonymous default.
    grader_name = "anon"
    blind_mode = False
    compare_mode = False
    token = os.getenv("EVAL_ADMIN_TOKEN", "")
    if request is not None:
        grader_name = request.cookies.get("grader", "anon")
        blind_mode = request.cookies.get("eval_blind", "") == "1"
        compare_mode = request.cookies.get("eval_compare", "") == "1"
        token = request.cookies.get("eval_admin") or request.query_params.get("token") or token

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


def _initials(name: str) -> str:
    """Derive 1- or 2-letter initials from a name. Falls back to 'AN'."""

    if not name or not name.strip():
        return "AN"
    parts = [p for p in name.replace("_", " ").replace("-", " ").split() if p]
    if not parts:
        return name[:2].upper()
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()


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


# ── Internals (patched in tests) ──────────────────────────────────────────────


def _eval_root(cfg) -> Path:
    """Resolve the eval data root from config, falling back to ``data/eval``.

    Tests ``monkeypatch.setattr`` this function to redirect writes to
    a tmp dir. The runtime path goes through ``cfg.eval.root`` (new
    config block added in Task 30) when present; otherwise it derives
    a path under ``cfg.paths.data_dir`` to stay inside the project
    sandbox; otherwise the literal ``"data/eval"`` fallback.
    """

    eval_cfg = getattr(cfg, "eval", None)
    if eval_cfg is not None:
        root = getattr(eval_cfg, "root", None)
        if root:
            return Path(root)
    paths = getattr(cfg, "paths", None)
    if paths is not None:
        data_dir = getattr(paths, "data_dir", None)
        if data_dir:
            return Path(data_dir) / "eval"
    return Path("data/eval")


def _eval_run_id(cfg) -> str:
    """Resolve the current run id from config, falling back to ``"default"``."""

    eval_cfg = getattr(cfg, "eval", None)
    if eval_cfg is not None:
        run_id = getattr(eval_cfg, "run_id", None)
        if run_id:
            return str(run_id)
    return "default"


def _load_queries(root: Path, run_id: str) -> list[dict[str, Any]]:
    """Load the curated query list for the run. Empty when missing.

    Task 33 ships the seeded queries file; until then this returns
    ``[]`` and the /eval page renders an empty-state queue.
    """

    queries_path = root / f"{run_id}.queries.json"
    if not queries_path.exists():
        return []
    try:
        return json.loads(queries_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []


def _grades_by_query(loaded: LoadedRun) -> dict[str, list[Grade]]:
    """Group LoadedRun.grades by query_id."""

    out: dict[str, list[Grade]] = {}
    for (qid, _scene_id), entry in loaded.grades.items():
        out.setdefault(qid, []).append(entry.grade)
    return out


def _grades_for_query(loaded: LoadedRun, query_id: str) -> list[Grade]:
    """Return the grade list for a single query (any scene id)."""

    return [entry.grade for (qid, _scene_id), entry in loaded.grades.items() if qid == query_id]


def _annotator_summary(loaded: LoadedRun) -> tuple[int, float]:
    """Return (distinct annotator count, κ between top-2 graders).

    κ is computed only on (query, scene) pairs that BOTH top graders
    rated. With < 2 graders or no overlap, κ = 0.0. With only one
    annotator on file, the /eval header simply shows "1 annotator"
    and no κ pill.
    """

    by_grader: dict[str, dict[tuple[str, str], Grade]] = {}
    for key, entry in loaded.grades.items():
        by_grader.setdefault(entry.grader, {})[key] = entry.grade

    annotator_count = len(by_grader)
    if annotator_count < 2:
        return annotator_count, 0.0

    ranked = sorted(by_grader.items(), key=lambda kv: -len(kv[1]))
    (_, a_grades), (_, b_grades) = ranked[0], ranked[1]
    shared = a_grades.keys() & b_grades.keys()
    if not shared:
        return annotator_count, 0.0

    ordered = sorted(shared)
    a_list = [a_grades[k] for k in ordered]
    b_list = [b_grades[k] for k in ordered]
    return annotator_count, cohen_kappa(a_list, b_list)
