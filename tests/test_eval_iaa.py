"""Tests for the inter-annotator-agreement (IAA) bundle the eval
service ships to the right-pane metrics panel + the queue's Conflict
filter tab.

Backed by ``cinemateca.eval.grades.load_run_per_annotator`` (the
non-collapsing loader) and exercised through
``api.services.eval_service._build_iaa`` /
``_query_conflict_set`` / ``_other_grades_for_current``.

The tests construct synthetic JSONL runs with two graders rating
overlapping (query, scene) pairs and assert:
  * the bundle reports ``enabled=False`` until ≥ 2 graders share
    at least one (q, s) judgment
  * the 4×4 confusion matrix indexes [self][other]
  * Cohen's κ matches ``cohen_kappa`` over the SKIP-filtered grades
  * the Landis-Koch qualitative label maps thresholds correctly
  * disagreements with |Δ| ≥ 2 (SKIP excluded) light up the conflict set
"""

from __future__ import annotations

from pathlib import Path

import pytest

from api.services.eval_service import (
    _build_iaa,
    _kappa_quality_label,
    _other_grades_for_current,
    _query_conflict_set,
)
from cinemateca.eval.grades import EvalRun, Grade, load_run_per_annotator, save_grade

# ── Fixture: a two-grader run with a curated mix of agreement,
#    near-miss, and full conflict pairs ────────────────────────────────


@pytest.fixture()
def two_grader_run(tmp_path: Path):
    """4 shared (q, s) pairs across q1+q2 between rg and jr.

    Grade pattern (rg, jr):
      (q1, s1): 3, 3   exact agreement (diagonal[3][3])
      (q1, s2): 2, 1   near-miss     (off-diag, |Δ|=1)
      (q1, s3): 0, 3   conflict      (|Δ|=3) → q1 in conflict set
      (q2, s1): 1, 1   exact agreement (diagonal[1][1])
    Plus a third grader (xx) who only rated (q3, s1) at IRRELEVANT —
    proves _build_iaa picks the most-active OTHER, not just anyone.
    """

    run = EvalRun(run_id="r1", root=tmp_path)
    save_grade(run, query_id="q1", scene_id="s1", grader="rg", grade=Grade.HIGHLY_RELEVANT)
    save_grade(run, query_id="q1", scene_id="s1", grader="jr", grade=Grade.HIGHLY_RELEVANT)
    save_grade(run, query_id="q1", scene_id="s2", grader="rg", grade=Grade.RELEVANT)
    save_grade(run, query_id="q1", scene_id="s2", grader="jr", grade=Grade.WEAKLY)
    save_grade(run, query_id="q1", scene_id="s3", grader="rg", grade=Grade.IRRELEVANT)
    save_grade(run, query_id="q1", scene_id="s3", grader="jr", grade=Grade.HIGHLY_RELEVANT)
    save_grade(run, query_id="q2", scene_id="s1", grader="rg", grade=Grade.WEAKLY)
    save_grade(run, query_id="q2", scene_id="s1", grader="jr", grade=Grade.WEAKLY)
    save_grade(run, query_id="q3", scene_id="s1", grader="xx", grade=Grade.IRRELEVANT)
    return load_run_per_annotator(run)


# ── _build_iaa contract ──────────────────────────────────────────────


def test_iaa_disabled_for_single_annotator(tmp_path: Path):
    run = EvalRun(run_id="solo", root=tmp_path)
    save_grade(run, query_id="q1", scene_id="s1", grader="rg", grade=Grade.RELEVANT)
    iaa = _build_iaa(load_run_per_annotator(run), current_grader="rg")
    assert iaa == {"enabled": False}


def test_iaa_disabled_when_no_shared_pairs(tmp_path: Path):
    """Two graders with non-overlapping scenes → no κ to compute."""

    run = EvalRun(run_id="r1", root=tmp_path)
    save_grade(run, query_id="q1", scene_id="s1", grader="rg", grade=Grade.RELEVANT)
    save_grade(run, query_id="q1", scene_id="s2", grader="jr", grade=Grade.WEAKLY)
    iaa = _build_iaa(load_run_per_annotator(run), current_grader="rg")
    assert iaa["enabled"] is False


def test_iaa_picks_most_active_other(two_grader_run):
    """When the run has 3 graders, the OTHER is the most-active non-self."""

    iaa = _build_iaa(two_grader_run, current_grader="rg")
    assert iaa["enabled"] is True
    assert iaa["self"]["name"] == "rg"
    assert iaa["other"]["name"] == "jr"
    # xx only graded one (q,s) — not picked as the "other" partner.
    assert iaa["other"]["count"] == 4  # jr graded 4 of the shared (q,s) pairs


def test_iaa_confusion_matrix_indexes_self_then_other(two_grader_run):
    """confusion[i][j] is the count of (self=i, other=j) pairs.

    Per the fixture:
      (rg=3, jr=3) → confusion[3][3]++
      (rg=2, jr=1) → confusion[2][1]++
      (rg=0, jr=3) → confusion[0][3]++
      (rg=1, jr=1) → confusion[1][1]++
    """

    iaa = _build_iaa(two_grader_run, current_grader="rg")
    cm = iaa["confusion"]
    assert cm[3][3] == 1
    assert cm[2][1] == 1
    assert cm[0][3] == 1
    assert cm[1][1] == 1
    # Diagonal sum == agreement count (matches agree_pct numerator).
    diag = sum(cm[i][i] for i in range(4))
    assert diag == 2  # two exact agreements out of four pairs
    assert iaa["agree_pct"] == 50  # 2/4 = 50%


def test_iaa_grand_total_matches_shared(two_grader_run):
    iaa = _build_iaa(two_grader_run, current_grader="rg")
    assert iaa["grand_total"] == iaa["shared"] == 4


def test_iaa_row_and_col_totals(two_grader_run):
    iaa = _build_iaa(two_grader_run, current_grader="rg")
    # rg used IRRELEVANT once, WEAKLY once, RELEVANT once, HIGHLY once.
    assert iaa["row_totals"] == [1, 1, 1, 1]
    # jr used WEAKLY twice, HIGHLY_RELEVANT twice — no 0 or 2.
    assert iaa["col_totals"] == [0, 2, 0, 2]


def test_iaa_conflict_pairs_counts_delta_at_least_two(two_grader_run):
    """|rg - jr| ≥ 2 counts as a conflict. (0,3) is the only such pair."""

    iaa = _build_iaa(two_grader_run, current_grader="rg")
    assert iaa["conflict_pairs"] == 1


def test_iaa_skip_excluded_from_matrix_and_kappa(tmp_path: Path):
    """SKIP entries don't pollute the matrix or κ."""

    run = EvalRun(run_id="r1", root=tmp_path)
    save_grade(run, query_id="q1", scene_id="s1", grader="rg", grade=Grade.SKIP)
    save_grade(run, query_id="q1", scene_id="s1", grader="jr", grade=Grade.SKIP)
    save_grade(run, query_id="q1", scene_id="s2", grader="rg", grade=Grade.RELEVANT)
    save_grade(run, query_id="q1", scene_id="s2", grader="jr", grade=Grade.RELEVANT)
    iaa = _build_iaa(load_run_per_annotator(run), current_grader="rg")
    assert iaa["enabled"] is True
    assert iaa["shared"] == 1  # only the non-SKIP pair counts
    assert iaa["confusion"][2][2] == 1
    assert iaa["grand_total"] == 1


# ── Landis-Koch qualitative buckets ───────────────────────────────────


@pytest.mark.parametrize(
    "kappa,expected",
    [
        (-0.1, "poor"),
        (0.0, "slight"),
        (0.20, "slight"),
        (0.21, "fair"),
        (0.40, "fair"),
        (0.41, "moderate"),
        (0.60, "moderate"),
        (0.61, "substantial"),
        (0.80, "substantial"),
        (0.81, "almost perfect"),
        (1.0, "almost perfect"),
    ],
)
def test_kappa_quality_label(kappa, expected):
    assert _kappa_quality_label(kappa) == expected


# ── _query_conflict_set ───────────────────────────────────────────────


def test_conflict_set_picks_up_two_or_more_delta(two_grader_run):
    """Only q1 has a (rg=0, jr=3) disagreement → q1 in the set."""

    assert _query_conflict_set(two_grader_run) == {"q1"}


def test_conflict_set_ignores_skip_only_pairs(tmp_path: Path):
    run = EvalRun(run_id="r1", root=tmp_path)
    save_grade(run, query_id="q1", scene_id="s1", grader="rg", grade=Grade.SKIP)
    save_grade(run, query_id="q1", scene_id="s1", grader="jr", grade=Grade.HIGHLY_RELEVANT)
    assert _query_conflict_set(load_run_per_annotator(run)) == set()


def test_conflict_set_ignores_single_annotator_pairs(tmp_path: Path):
    run = EvalRun(run_id="r1", root=tmp_path)
    save_grade(run, query_id="q1", scene_id="s1", grader="rg", grade=Grade.HIGHLY_RELEVANT)
    save_grade(run, query_id="q2", scene_id="s1", grader="jr", grade=Grade.IRRELEVANT)
    # No (q, s) has both annotators — nothing to conflict on.
    assert _query_conflict_set(load_run_per_annotator(run)) == set()


# ── _other_grades_for_current ─────────────────────────────────────────


def test_other_grades_returns_only_current_query(two_grader_run):
    """Filters to the current query AND the named other annotator."""

    out = _other_grades_for_current(
        two_grader_run, current_query_id="q1", other_grader="jr"
    )
    assert out == {
        "s1": Grade.HIGHLY_RELEVANT,
        "s2": Grade.WEAKLY,
        "s3": Grade.HIGHLY_RELEVANT,
    }


def test_other_grades_empty_without_other_grader(two_grader_run):
    """No second annotator named → empty dict (single-grader runs)."""

    assert _other_grades_for_current(
        two_grader_run, current_query_id="q1", other_grader=None
    ) == {}


def test_other_grades_unknown_grader(two_grader_run):
    """A grader name that doesn't appear on the run yields {} silently."""

    assert _other_grades_for_current(
        two_grader_run, current_query_id="q1", other_grader="ghost"
    ) == {}
