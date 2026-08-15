"""Everything in the /eval page context that depends on the current query.

Split out of the package ``__init__`` to keep it inside the 250-line
``api/services/**`` budget. The seam is the current query: this module is
handed one already-resolved query plus the loaded run, and it never calls
``_eval_root`` / ``_eval_run_id``, so the fixtures that monkeypatch those two
names on the package module still reach every caller of them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kuaa.eval.grader_metrics import (
    grades_for_current_grader,
    histogram,
    inversions,
    ndcg_at_k,
    precision_at_k,
)
from kuaa.eval.grader_metrics import (
    other_grades_for_current as _other_grades_for_current,
)
from kuaa.eval.grades import Grade
from kuaa.eval.grades import grades_for_query as _grades_for_query
from kuaa.eval.slates import hydrate_rows as _hydrate_rows

#: Zeroed metrics, so the template renders on a query that has no grades yet.
_EMPTY_METRICS: dict[str, Any] = {
    "p_at_3": 0.0,
    "p_at_5": 0.0,
    "ndcg_at_5": 0.0,
    "inversions": 0,
    "histogram": {},
}


def build_current_query_view(
    current_query: dict[str, Any] | None,
    *,
    cfg: Any,
    loaded: Any,
    per_annotator: Any,
    iaa: dict[str, Any],
    grader_name: str,
) -> dict[str, Any]:
    """Build the current-query slice of the /eval context.

    Hydrates ``current_query["results"]`` in place as a side effect. Returns
    the four keys the page context needs: ``grades_for_current``,
    ``grades_for_current_other``, ``metrics``, ``result_count``.
    """
    grades_for_current: dict[str, Grade] = {}
    grades_for_current_other: dict[str, Grade] = {}
    metrics = dict(_EMPTY_METRICS)
    result_count = 0

    if current_query is None:
        return {
            "grades_for_current": grades_for_current,
            "grades_for_current_other": grades_for_current_other,
            "metrics": metrics,
            "result_count": result_count,
        }

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
        # Slates persist provenance only (film_slug / scene_id / pool);
        # presentation is re-read from per-film metadata here. Only the
        # current query is hydrated — rows.html is the sole consumer, and
        # doing all of them would re-read every film's metadata per render.
        # A fat slate written before thinning passes through untouched.
        library_dir = Path(
            getattr(getattr(cfg, "paths", None), "library_dir", None) or "data/library"
        )
        current_query["results"] = _hydrate_rows(results, cfg=cfg, library_dir=library_dir)

    return {
        "grades_for_current": grades_for_current,
        "grades_for_current_other": grades_for_current_other,
        "metrics": metrics,
        "result_count": result_count,
    }
