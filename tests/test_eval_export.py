"""Tests for export_run — collapses append-only JSONL to graded structure.

Deliverable: E5 (agent part) — grade export.

Two cases:
  1. Basic structure + last-write-wins collapse (a re-graded pair reflects
     the LATEST grade, not the earlier one).
  2. Summary counts are correct for distinct pairs / queries / graders.
"""

from __future__ import annotations

from pathlib import Path

from cinemateca.eval.grades import EvalRun, Grade, save_grade
from cinemateca.eval.grades import export_run


def test_export_run_collapses_to_graded_structure(tmp_path: Path) -> None:
    """export_run collapses JSONL; a re-graded pair reflects the LATEST grade."""

    run = EvalRun(run_id="t1", root=tmp_path)

    # q1/s1: graded IRRELEVANT first, then re-graded HIGHLY_RELEVANT (latest wins).
    save_grade(run, query_id="q1", scene_id="s1", grader="rg", grade=Grade.IRRELEVANT)
    save_grade(run, query_id="q1", scene_id="s1", grader="rg", grade=Grade.HIGHLY_RELEVANT)

    # q1/s2: graded once.
    save_grade(run, query_id="q1", scene_id="s2", grader="rg", grade=Grade.RELEVANT)

    # q2/s3: graded by a different annotator.
    save_grade(run, query_id="q2", scene_id="s3", grader="jr", grade=Grade.WEAKLY)

    result = export_run(run)

    # Top-level shape.
    assert result["run_id"] == "t1"
    assert "grades" in result
    assert "summary" in result

    # q1/s1 must show the LATEST grade (HIGHLY_RELEVANT = 3), not IRRELEVANT (0).
    assert result["grades"]["q1"]["s1"] == int(Grade.HIGHLY_RELEVANT), (
        "expected latest grade (HIGHLY_RELEVANT=3), "
        f"got {result['grades']['q1']['s1']!r}"
    )

    # q1/s2 grade.
    assert result["grades"]["q1"]["s2"] == int(Grade.RELEVANT)

    # q2/s3 grade.
    assert result["grades"]["q2"]["s3"] == int(Grade.WEAKLY)

    # Summary: 3 distinct (qid, sid) pairs (q1/s1, q1/s2, q2/s3).
    assert result["summary"]["distinct_pairs"] == 3, (
        f"expected 3 distinct pairs, got {result['summary']['distinct_pairs']}"
    )
    assert result["summary"]["queries"] == 2
    assert result["summary"]["graders"] == 2


def test_export_run_empty_when_no_file(tmp_path: Path) -> None:
    """An absent JSONL returns a valid zero-state structure."""

    run = EvalRun(run_id="never_written", root=tmp_path)
    result = export_run(run)

    assert result["run_id"] == "never_written"
    assert result["grades"] == {}
    assert result["summary"]["distinct_pairs"] == 0
    assert result["summary"]["queries"] == 0
    assert result["summary"]["graders"] == 0


def test_export_run_exported_from_package() -> None:
    """export_run is importable from cinemateca.eval (public surface check)."""

    from cinemateca.eval import export_run as _exported  # noqa: F401

    assert callable(_exported)
