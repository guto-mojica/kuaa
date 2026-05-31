"""Eval-set-builder grading metrics (Task 30).

Live-updating quality indicators on the /eval page right pane:

* ``precision_at_k`` — fraction of top-k items with grade >= RELEVANT
* ``ndcg_at_k`` — normalised DCG with ``2^g - 1`` gain (SKIP → 0)
* ``inversions`` — out-of-order pair count (sanity on ordering)
* ``histogram`` — count of each grade value (incl. SKIP)
* ``cohen_kappa`` — inter-annotator agreement κ between two grader lists

Module-path divergence from the plan
------------------------------------
The plan calls this ``cinemateca.eval.metrics``. That name is already
taken by ``cinemateca.eval.metrics`` for the *retrieval* metrics
(recall@K, MRR, graded-relevance nDCG). Reusing the name would shadow
the retrieval ``ndcg_at_k`` with a Grade-list signature that means
something else. Both surfaces coexist in their own modules; this file
covers the grading-UI quality indicators.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from cinemateca.eval.grades import Grade, GradeEntry


def precision_at_k(grades: list[Grade], k: int) -> float:
    """Fraction of top-k items with grade >= ``RELEVANT`` (i.e. >= 2).

    SKIP rows are excluded from both numerator AND denominator: a
    grader marking an item SKIP means "no opinion", not "irrelevant",
    so it should not bias the precision score in either direction.
    Empty input returns 0.0.
    """

    if k <= 0 or not grades:
        return 0.0
    top = [g for g in grades[:k] if g != Grade.SKIP]
    if not top:
        return 0.0
    relevant = sum(1 for g in top if int(g) >= int(Grade.RELEVANT))
    return relevant / len(top)


def _gain(g: Grade) -> float:
    """DCG gain function — ``2^g - 1`` with SKIP contributing 0."""

    if g == Grade.SKIP:
        return 0.0
    val = int(g)
    if val <= 0:
        return 0.0
    return float((1 << val) - 1)  # 2**val - 1, integer-exact


def _discount(rank_zero_indexed: int) -> float:
    """Standard DCG discount: ``1 / log2(rank + 2)`` for zero-indexed rank."""

    return 1.0 / math.log2(rank_zero_indexed + 2)


def ndcg_at_k(grades: list[Grade], k: int) -> float:
    """Normalised DCG@k using ``2^g - 1`` gain.

    The ideal ranking is obtained by sorting the SAME top-k slice by
    descending grade (SKIP last) — this measures how well the system's
    order matches the best achievable order *for the items the grader
    saw*, not against an external ideal. ``idcg == 0`` (no positive
    grades in the slice) → 0.0.
    """

    if k <= 0 or not grades:
        return 0.0
    top = list(grades[:k])
    dcg = sum(_gain(g) * _discount(i) for i, g in enumerate(top))
    ideal = sorted(
        top,
        key=lambda g: (-1, 0) if g == Grade.SKIP else (-int(g), 0),
    )
    idcg = sum(_gain(g) * _discount(i) for i, g in enumerate(ideal))
    if idcg <= 0.0:
        return 0.0
    return dcg / idcg


def inversions(grades: list[Grade]) -> int:
    """Count out-of-order ranked pairs.

    For each i < j, count ``grades[i] < grades[j]`` (the lower-ranked
    item has a higher grade — an inversion). SKIP rows are ignored;
    they neither produce nor receive inversions. O(n^2); rankings on
    the /eval page are short (k <= 10) so this is fine.
    """

    n = 0
    for i in range(len(grades)):
        if grades[i] == Grade.SKIP:
            continue
        for j in range(i + 1, len(grades)):
            if grades[j] == Grade.SKIP:
                continue
            if int(grades[i]) < int(grades[j]):
                n += 1
    return n


def histogram(grades: list[Grade]) -> dict[int, int]:
    """Count of each grade value in the input, keyed by ``int(grade)``.

    Includes SKIP (key ``-1``) when present. Grades not in the input
    do not appear in the result (callers shouldn't assume the full
    -1..3 range; use ``hist.get(g, 0)`` if needed).
    """

    return dict(Counter(int(g) for g in grades))


def cohen_kappa(a: list[Grade], b: list[Grade]) -> float:
    """Cohen's κ between two equal-length grader lists.

    Returns 0.0 when the inputs are empty or lengths disagree (κ is
    undefined in those cases; surfacing 0 keeps the metrics endpoint
    a safe stub when only one annotator has graded).

    Formula::

        κ = (Po - Pe) / (1 - Pe)

    where ``Po`` is observed agreement and ``Pe`` is chance agreement
    computed across the union of category labels. When ``Pe`` is 1.0
    (one grader always uses the same category and matches every time),
    the formula is degenerate; return 1.0 if Po == 1 else 0.0.
    """

    if not a or len(a) != len(b):
        return 0.0

    n = len(a)
    agreements = sum(1 for x, y in zip(a, b) if x == y)
    po = agreements / n

    categories = {int(g) for g in a} | {int(g) for g in b}
    pe = 0.0
    for c in categories:
        pa = sum(1 for x in a if int(x) == c) / n
        pb = sum(1 for y in b if int(y) == c) / n
        pe += pa * pb

    if math.isclose(pe, 1.0, abs_tol=1e-12):
        return 1.0 if math.isclose(po, 1.0, abs_tol=1e-12) else 0.0
    return (po - pe) / (1.0 - pe)


# ── IAA + annotator helpers ───────────────────────────────────────────────────

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


def initials(name: str) -> str:
    """Derive 1- or 2-letter initials from a name. Falls back to 'AN'."""

    if not name or not name.strip():
        return "AN"
    parts = [p for p in name.replace("_", " ").replace("-", " ").split() if p]
    if not parts:
        return name[:2].upper()
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()


def kappa_quality_label(kappa: float) -> str:
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


def grader_initials(name: str) -> str:
    """Local mirror of ``initials`` for grader pills inside IAA blocks.

    Kept as a named alias so the existing ``initials``
    contract (anon → 'AN') stays untouched.
    """

    return initials(name)


def annotator_summary(
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


def build_iaa(
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
            "initials": grader_initials(self_name),
            "count": grader_counts[self_name],
        },
        "other": {
            "name": other_name,
            "initials": grader_initials(other_name),
            "count": grader_counts[other_name],
        },
        "shared": shared,
        "kappa": kappa,
        "agree_pct": agree_pct,
        "quality_label": kappa_quality_label(kappa),
        "confusion": confusion,
        "row_totals": row_totals,
        "col_totals": col_totals,
        "grand_total": grand_total,
        "conflict_pairs": conflict_pairs,
    }


def other_grades_for_current(
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


def grades_for_current_grader(
    per_annotator: dict[tuple[str, str], dict[str, GradeEntry]],
    loaded_grades: dict[tuple[str, str], GradeEntry],
    *,
    current_query_id: str,
    grader_name: str,
) -> dict[str, Grade]:
    """``scene_id -> Grade`` for the CURRENT grader on the current query.

    Drives the ``.gb`` chip render in rows.html: it must show THIS grader's
    grade, not the last-write-wins reduce of ``loaded_grades`` (which would
    surface the other annotator's grade in a multi-grader run and make every
    disagree-by-≥2 chip vanish). Reads from ``per_annotator`` keyed by
    ``grader_name``; when the current grader has no record on a scene but
    someone else does, it falls back to the collapsed ``loaded_grades`` entry
    so the row isn't ungraded-looking (preserving the prior single-grader
    semantics).

    The mirror of :func:`other_grades_for_current` for the active grader.
    """

    out: dict[str, Grade] = {}
    for (qid, scene_id), by_who in per_annotator.items():
        if qid != current_query_id:
            continue
        if grader_name in by_who:
            out[str(scene_id)] = by_who[grader_name].grade
        elif by_who:
            canonical = loaded_grades.get((qid, scene_id))
            if canonical is not None:
                out[str(scene_id)] = canonical.grade
    return out


def query_conflict_set(
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
