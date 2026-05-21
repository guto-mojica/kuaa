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

from cinemateca.eval.grades import Grade


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
