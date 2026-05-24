"""Tests for the Eval-set-builder grade persistence (Task 30).

Covers the JSONL append-only log format, round-trip via ``load_run``, the
``Grade`` enum value contract, the "overwrite keeps history" semantics, and
the SKIP grade (no-relevance opinion).
"""

from __future__ import annotations

from pathlib import Path

from cinemateca.eval.grades import (
    EvalRun,
    Grade,
    GradeEntry,
    load_run,
    save_grade,
)


def test_grade_enum_values():
    """Numeric values are part of the persistence contract."""

    assert int(Grade.IRRELEVANT) == 0
    assert int(Grade.WEAKLY) == 1
    assert int(Grade.RELEVANT) == 2
    assert int(Grade.HIGHLY_RELEVANT) == 3
    assert int(Grade.SKIP) == -1


def test_save_and_load_grade(tmp_path: Path):
    run = EvalRun(run_id="r1", root=tmp_path)
    save_grade(
        run,
        query_id="q1",
        scene_id="jeca/1",
        grader="rg",
        grade=Grade.RELEVANT,
    )

    loaded = load_run(run)
    assert ("q1", "jeca/1") in loaded.grades
    entry = loaded.grades[("q1", "jeca/1")]
    assert isinstance(entry, GradeEntry)
    assert entry.grader == "rg"
    assert entry.grade == Grade.RELEVANT
    assert entry.query_id == "q1"
    assert entry.scene_id == "jeca/1"
    # Timestamp string is set on save (ISO 8601 UTC).
    assert entry.ts


def test_grade_overwrite_keeps_history(tmp_path: Path):
    """Append-only log: both writes preserved, latest wins on load."""

    run = EvalRun(run_id="r1", root=tmp_path)
    save_grade(
        run,
        query_id="q1",
        scene_id="jeca/1",
        grader="rg",
        grade=Grade.IRRELEVANT,
    )
    save_grade(
        run,
        query_id="q1",
        scene_id="jeca/1",
        grader="rg",
        grade=Grade.HIGHLY_RELEVANT,
    )

    lines = (tmp_path / "r1.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2

    loaded = load_run(run)
    assert loaded.grades[("q1", "jeca/1")].grade == Grade.HIGHLY_RELEVANT


def test_skip_grade_persists(tmp_path: Path):
    """SKIP (-1) is a first-class grade and round-trips."""

    run = EvalRun(run_id="r1", root=tmp_path)
    save_grade(
        run,
        query_id="q1",
        scene_id="jeca/1",
        grader="rg",
        grade=Grade.SKIP,
    )

    loaded = load_run(run)
    assert loaded.grades[("q1", "jeca/1")].grade == Grade.SKIP


def test_load_run_empty_when_no_file(tmp_path: Path):
    run = EvalRun(run_id="never_written", root=tmp_path)
    loaded = load_run(run)
    assert loaded.run_id == "never_written"
    assert loaded.grades == {}


def test_save_grade_creates_root_dir(tmp_path: Path):
    """save_grade is robust to a missing root directory."""

    root = tmp_path / "deep" / "nested"
    run = EvalRun(run_id="r1", root=root)
    save_grade(
        run,
        query_id="q1",
        scene_id="jeca/1",
        grader="rg",
        grade=Grade.RELEVANT,
    )
    assert (root / "r1.jsonl").exists()


def test_multiple_query_scene_pairs(tmp_path: Path):
    """Different (query, scene) keys are independent."""

    run = EvalRun(run_id="r1", root=tmp_path)
    save_grade(run, query_id="q1", scene_id="jeca/1", grader="rg", grade=Grade.RELEVANT)
    save_grade(run, query_id="q1", scene_id="jeca/2", grader="rg", grade=Grade.WEAKLY)
    save_grade(
        run,
        query_id="q2",
        scene_id="jeca/1",
        grader="rg",
        grade=Grade.IRRELEVANT,
    )

    loaded = load_run(run)
    assert len(loaded.grades) == 3
    assert loaded.grades[("q1", "jeca/1")].grade == Grade.RELEVANT
    assert loaded.grades[("q1", "jeca/2")].grade == Grade.WEAKLY
    assert loaded.grades[("q2", "jeca/1")].grade == Grade.IRRELEVANT


# ── load_run_per_annotator: multi-annotator preservation ─────────────────────


def test_load_run_per_annotator_keeps_each_grader(tmp_path: Path):
    """Two annotators on the same (q,s) — both grades survive the load.

    The default ``load_run`` collapses across graders (last-write-wins),
    which makes inter-annotator-agreement math impossible. The new
    ``load_run_per_annotator`` view groups by grader so the IAA panel
    in the right pane can compare them.
    """

    from cinemateca.eval.grades import load_run_per_annotator

    run = EvalRun(run_id="r1", root=tmp_path)
    save_grade(run, query_id="q1", scene_id="s1", grader="rg", grade=Grade.RELEVANT)
    save_grade(
        run, query_id="q1", scene_id="s1", grader="jr", grade=Grade.HIGHLY_RELEVANT
    )

    per_annot = load_run_per_annotator(run)
    assert ("q1", "s1") in per_annot
    bucket = per_annot[("q1", "s1")]
    assert set(bucket) == {"rg", "jr"}
    assert bucket["rg"].grade == Grade.RELEVANT
    assert bucket["jr"].grade == Grade.HIGHLY_RELEVANT


def test_load_run_per_annotator_regrade_supersedes_self(tmp_path: Path):
    """A regrade by the same grader overwrites their earlier vote only."""

    from cinemateca.eval.grades import load_run_per_annotator

    run = EvalRun(run_id="r1", root=tmp_path)
    save_grade(run, query_id="q1", scene_id="s1", grader="rg", grade=Grade.WEAKLY)
    save_grade(run, query_id="q1", scene_id="s1", grader="jr", grade=Grade.RELEVANT)
    save_grade(run, query_id="q1", scene_id="s1", grader="rg", grade=Grade.HIGHLY_RELEVANT)

    per_annot = load_run_per_annotator(run)
    bucket = per_annot[("q1", "s1")]
    # rg's WEAKLY was superseded by their later HIGHLY_RELEVANT —
    # jr's RELEVANT stays put even though it was older than rg's regrade.
    assert bucket["rg"].grade == Grade.HIGHLY_RELEVANT
    assert bucket["jr"].grade == Grade.RELEVANT


def test_load_run_per_annotator_missing_file(tmp_path: Path):
    """Missing JSONL → empty dict, mirroring ``load_run``."""

    from cinemateca.eval.grades import load_run_per_annotator

    run = EvalRun(run_id="never_written", root=tmp_path)
    assert load_run_per_annotator(run) == {}
