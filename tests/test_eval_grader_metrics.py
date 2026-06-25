"""Tests for the Eval-set-builder grading metrics (Task 30).

Covers precision@K, nDCG@K, ranking inversions, grade histogram, and
Cohen's kappa between two graders. These are the live-updating
quality indicators surfaced on the /eval page right pane.

Module path note
----------------
The plan calls this module ``kuaa.eval.metrics``, but that name
is already taken by the *retrieval* metrics (recall@K, MRR, graded
nDCG with a relevance-dict signature). Reusing the same module would
silently shadow `ndcg_at_k` with a list-of-Grade signature that means
something quite different. The grading-side metrics therefore live in
``kuaa.eval.grader_metrics`` (and tests in
``tests/test_eval_grader_metrics.py``), keeping both call surfaces
ergonomic and the retrieval-metrics tests green.
"""

from __future__ import annotations

import math

from kuaa.eval.grader_metrics import (
    cohen_kappa,
    histogram,
    inversions,
    ndcg_at_k,
    precision_at_k,
)
from kuaa.eval.grades import Grade


def test_precision_at_k_basic():
    """Counts grades >= RELEVANT (2) in the top-k slice."""

    grades = [
        Grade.HIGHLY_RELEVANT,
        Grade.RELEVANT,
        Grade.IRRELEVANT,
        Grade.WEAKLY,
        Grade.RELEVANT,
    ]
    # P@5 = 3 / 5 (HR, R, R are >= 2)
    assert precision_at_k(grades, k=5) == 3 / 5
    # P@3 = 2 / 3 (HR, R are >= 2; IR is not)
    assert precision_at_k(grades, k=3) == 2 / 3


def test_precision_at_k_excludes_skip_from_denominator():
    """SKIP is "no opinion", neither numerator nor denominator."""

    grades = [Grade.RELEVANT, Grade.SKIP, Grade.IRRELEVANT]
    # Only 2 graded items in top-3 → 1 relevant / 2 = 0.5
    assert precision_at_k(grades, k=3) == 0.5


def test_precision_at_k_empty():
    assert precision_at_k([], k=5) == 0.0


def test_ndcg_at_k_perfect():
    """Ranking already in ideal order → nDCG = 1.0."""

    grades = [
        Grade.HIGHLY_RELEVANT,
        Grade.RELEVANT,
        Grade.WEAKLY,
        Grade.IRRELEVANT,
        Grade.IRRELEVANT,
    ]
    assert math.isclose(ndcg_at_k(grades, k=5), 1.0, abs_tol=1e-6)


def test_ndcg_at_k_worst_then_best():
    """Inverted ranking is less than ideal but > 0."""

    # All-zero grades return 0 (idcg = 0 floor)
    zeros = [Grade.IRRELEVANT, Grade.IRRELEVANT]
    assert ndcg_at_k(zeros, k=2) == 0.0

    # Reversed-perfect: ideal would be (3,2,1) but we got (1,2,3)
    grades = [Grade.WEAKLY, Grade.RELEVANT, Grade.HIGHLY_RELEVANT]
    score = ndcg_at_k(grades, k=3)
    assert 0.0 < score < 1.0


def test_inversions_counts_out_of_order_pairs():
    """Inversion: rank-i comes BEFORE rank-j but grade-i < grade-j."""

    # Rank 0 has grade 2 (RELEVANT); rank 1 has grade 3 (HR) → 1 inversion
    grades = [Grade.RELEVANT, Grade.HIGHLY_RELEVANT]
    assert inversions(grades) == 1

    # Already in order — no inversions
    grades = [Grade.HIGHLY_RELEVANT, Grade.RELEVANT, Grade.WEAKLY]
    assert inversions(grades) == 0

    # SKIP rows are ignored
    grades = [Grade.RELEVANT, Grade.SKIP, Grade.HIGHLY_RELEVANT]
    assert inversions(grades) == 1


def test_histogram_counts_each_grade_including_skip():
    grades = [Grade.IRRELEVANT, Grade.IRRELEVANT, Grade.RELEVANT, Grade.SKIP]
    h = histogram(grades)
    assert h[0] == 2
    assert h[2] == 1
    assert h[-1] == 1
    # WEAKLY (1) absent
    assert h.get(1, 0) == 0


def test_cohen_kappa_perfect_agreement():
    a = [Grade.RELEVANT, Grade.RELEVANT, Grade.IRRELEVANT]
    b = [Grade.RELEVANT, Grade.RELEVANT, Grade.IRRELEVANT]
    assert math.isclose(cohen_kappa(a, b), 1.0, abs_tol=1e-6)


def test_cohen_kappa_chance_agreement_is_low():
    """Two graders disagreeing on half the items → κ near 0."""

    a = [Grade.RELEVANT, Grade.RELEVANT, Grade.IRRELEVANT, Grade.IRRELEVANT]
    b = [Grade.RELEVANT, Grade.IRRELEVANT, Grade.RELEVANT, Grade.IRRELEVANT]
    k = cohen_kappa(a, b)
    # Po = 0.5, Pe = 0.5 → κ = 0 exactly
    assert math.isclose(k, 0.0, abs_tol=1e-6)


def test_cohen_kappa_complete_disagreement_negative():
    """Always-opposite ratings → κ < 0."""

    a = [Grade.RELEVANT, Grade.IRRELEVANT]
    b = [Grade.IRRELEVANT, Grade.RELEVANT]
    assert cohen_kappa(a, b) < 0.0


def test_cohen_kappa_empty_or_mismatched_length():
    assert cohen_kappa([], []) == 0.0
    assert cohen_kappa([Grade.RELEVANT], []) == 0.0
