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
from collections import Counter
from typing import Any

from cinemateca.eval.datasets import load_queries as _load_queries  # noqa: F401
from cinemateca.eval.grader_metrics import (
    cohen_kappa,
    histogram,
    inversions,
    ndcg_at_k,
    precision_at_k,
)
from cinemateca.eval.grades import (
    EvalRun,
    Grade,
    GradeEntry,
    LoadedRun,
    grades_by_query as _grades_by_query,  # noqa: F401
    grades_for_query as _grades_for_query,  # noqa: F401
    load_run,
    load_run_per_annotator,
)
from cinemateca.eval.paths import (  # noqa: F401
    eval_root as _eval_root,
    eval_run_id as _eval_run_id,
)

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
# _eval_root, _eval_run_id, _load_queries, _grades_by_query, _grades_for_query
# are re-exported from cinemateca.eval.{paths,datasets,grades} at the top of
# this file. The ``as _underscored`` aliases preserve the module-level names
# that test fixtures monkeypatch via ``monkeypatch.setattr(eval_service, "_eval_root", ...)``.


def _annotator_summary(
    per_annotator: dict[tuple[str, str], dict[str, GradeEntry]],
) -> tuple[int, float]:
    """Return (distinct annotator count, κ between top-2 graders).

    Reads the per-annotator-preserving view from
    ``load_run_per_annotator`` so a re-grade by one annotator doesn't
    erase the other annotator's vote on the same (q, s) — the bug the
    earlier LoadedRun-only path had.

    κ is computed only on (query, scene) pairs that BOTH top graders
    rated. With < 2 graders or no overlap, κ = 0.0. With only one
    annotator on file, the /eval header simply shows "1 annotator"
    and no κ pill.
    """

    by_grader: dict[str, dict[tuple[str, str], Grade]] = {}
    for key, by_who in per_annotator.items():
        for grader, entry in by_who.items():
            by_grader.setdefault(grader, {})[key] = entry.grade

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


# ── IAA panel + compare-mode helpers (Task 31 follow-on) ──────────────────────

# Landis & Koch (1977) qualitative thresholds for κ. The label appears
# next to the numeric value in the right-pane IAA card; the colour is
# chosen by the template based on whether κ ≥ 0.41 (warm) vs lower.
_KAPPA_QUALITY_THRESHOLDS = (
    (0.81, "almost perfect"),
    (0.61, "substantial"),
    (0.41, "moderate"),
    (0.21, "fair"),
    (0.00, "slight"),
)


def _kappa_quality_label(kappa: float) -> str:
    """Return the Landis-Koch qualitative bucket for a κ value.

    Negative κ (worse than chance) is reported as "poor" so the UI has
    a distinct label to render. Returns ``"slight"`` for κ ≥ 0 < 0.21.
    """

    if kappa < 0:
        return "poor"
    for threshold, label in _KAPPA_QUALITY_THRESHOLDS:
        if kappa >= threshold:
            return label
    return "slight"


def _grader_initials(name: str) -> str:
    """Local mirror of ``_initials`` for grader pills inside IAA blocks.

    Kept private to the IAA helpers so the existing ``_initials``
    contract (anon → 'AN') stays untouched.
    """

    return _initials(name)


def _build_iaa(
    per_annotator: dict[tuple[str, str], dict[str, GradeEntry]],
    *,
    current_grader: str,
) -> dict[str, Any]:
    """Build the inter-annotator-agreement bundle for the right pane.

    Compares ``current_grader`` against the most-active OTHER grader on
    file. When that pair has fewer than one shared (q, s) judgment, the
    bundle reports ``enabled=False`` and the template skips the panel.

    Returns:
        {
            "enabled": bool,                       # render the panel?
            "self":  {"name", "initials", "count"},
            "other": {"name", "initials", "count"},
            "shared": int,                         # # of (q,s) overlap
            "kappa": float,
            "agree_pct": int,                      # exact-match %
            "quality_label": str,                  # Landis-Koch bucket
            "confusion": [[c00,c01,c02,c03],       # 4×4 (excludes SKIP)
                           …],                     # rows = self grade
                                                   # cols = other grade
            "row_totals": [r0,r1,r2,r3],
            "col_totals": [c0,c1,c2,c3],
            "grand_total": int,                    # = sum of row totals
            "conflict_pairs": int,                 # |g_self − g_other| ≥ 2
        }
    """

    # Tally per-grader entry counts to pick "the other annotator".
    grader_counts: Counter[str] = Counter()
    for by_who in per_annotator.values():
        for grader in by_who:
            grader_counts[grader] += 1
    if not grader_counts:
        return {"enabled": False}

    # The "other" is the most-active grader that isn't us. When the
    # current grader has never written, we still compare against the
    # top two on file to keep the panel useful.
    others = [g for g in grader_counts if g != current_grader]
    if not others:
        return {"enabled": False}
    other_name = max(others, key=lambda g: grader_counts[g])

    # Self defaults to current_grader; when the grader has no entries
    # we fall back to the second-most-active person on file so the
    # panel still renders something meaningful (two arbitrary graders'
    # overlap rather than blank).
    if current_grader in grader_counts:
        self_name = current_grader
    else:
        ranked = grader_counts.most_common()
        if len(ranked) < 2:
            return {"enabled": False}
        self_name = ranked[0][0] if ranked[0][0] != other_name else ranked[1][0]

    # Walk the shared (q, s) pairs. SKIP grades are excluded from the
    # confusion matrix + κ — they are explicit "no opinion" markers,
    # not judgments, and treating them as a 5th category inflates
    # disagreement.
    confusion = [[0] * 4 for _ in range(4)]
    a_list: list[Grade] = []
    b_list: list[Grade] = []
    agreements = 0
    conflict_pairs = 0
    shared = 0
    for _key, by_who in per_annotator.items():
        if self_name not in by_who or other_name not in by_who:
            continue
        g_self = by_who[self_name].grade
        g_other = by_who[other_name].grade
        if g_self == Grade.SKIP or g_other == Grade.SKIP:
            continue
        shared += 1
        a_list.append(g_self)
        b_list.append(g_other)
        if g_self == g_other:
            agreements += 1
        if abs(int(g_self) - int(g_other)) >= 2:
            conflict_pairs += 1
        confusion[int(g_self)][int(g_other)] += 1

    if shared == 0:
        return {"enabled": False}

    kappa = cohen_kappa(a_list, b_list)
    agree_pct = round((agreements / shared) * 100) if shared else 0

    row_totals = [sum(row) for row in confusion]
    col_totals = [sum(confusion[r][c] for r in range(4)) for c in range(4)]
    grand_total = sum(row_totals)

    return {
        "enabled": True,
        "self": {
            "name": self_name,
            "initials": _grader_initials(self_name),
            "count": grader_counts[self_name],
        },
        "other": {
            "name": other_name,
            "initials": _grader_initials(other_name),
            "count": grader_counts[other_name],
        },
        "shared": shared,
        "kappa": kappa,
        "agree_pct": agree_pct,
        "quality_label": _kappa_quality_label(kappa),
        "confusion": confusion,
        "row_totals": row_totals,
        "col_totals": col_totals,
        "grand_total": grand_total,
        "conflict_pairs": conflict_pairs,
    }


def _other_grades_for_current(
    per_annotator: dict[tuple[str, str], dict[str, GradeEntry]],
    *,
    current_query_id: str,
    other_grader: str | None,
) -> dict[str, Grade]:
    """``scene_id -> Grade`` from the OTHER annotator on the current query.

    Used by rows.html when compare mode is on to render the second
    annotator's column. Returns an empty dict when there is no other
    grader on file (single-annotator runs degrade silently).
    """

    if not other_grader:
        return {}
    out: dict[str, Grade] = {}
    for (qid, sid), by_who in per_annotator.items():
        if qid != current_query_id:
            continue
        if other_grader in by_who:
            out[str(sid)] = by_who[other_grader].grade
    return out


def _query_conflict_set(
    per_annotator: dict[tuple[str, str], dict[str, GradeEntry]],
) -> set[str]:
    """Return query_ids that contain at least one |Δ| ≥ 2 disagreement.

    Drives the queue's "Conflict" filter tab and the ``conflict_count``
    badge. SKIP entries are ignored — only real grade-vs-grade
    disagreements count.
    """

    conflicting: set[str] = set()
    for (qid, _sid), by_who in per_annotator.items():
        if len(by_who) < 2:
            continue
        grades = [g.grade for g in by_who.values() if g.grade != Grade.SKIP]
        if len(grades) < 2:
            continue
        if max(int(g) for g in grades) - min(int(g) for g in grades) >= 2:
            conflicting.add(qid)
    return conflicting
