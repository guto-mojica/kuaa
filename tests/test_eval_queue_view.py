"""Tests for the /eval queue view model.

The queue pane once derived its own progress from ``grades_by_query``, which
groups every judgment in a run by query id regardless of grader or of which
pool the judgment was taken against. These pin the two properties that fixed:
a row's progress counts only the current pool's candidates, and its pips are
positional.
"""

from __future__ import annotations

from kuaa.eval.grades import Grade, GradeEntry, grade_scene_key
from kuaa.eval.queue_view import build_queue_rows


def _query(qid: str, pairs: list[tuple[str, int]], **extra) -> dict:
    """A pool query whose results are ``(film_slug, scene_id)`` in pool order."""
    return {
        "id": qid,
        "text": extra.get("text", "a query"),
        "lang": extra.get("lang", "pt"),
        "source": extra.get("source", "manual"),
        "results": [{"film_slug": slug, "scene_id": sid} for slug, sid in pairs],
    }


def _per_annotator(rows: list[tuple[str, str, str, Grade]]) -> dict:
    """``(query_id, scene_key, grader, grade)`` rows into the loader's shape."""
    out: dict[tuple[str, str], dict[str, GradeEntry]] = {}
    for qid, key, grader, grade in rows:
        entry = GradeEntry(query_id=qid, scene_id=key, grader=grader, grade=grade, ts="")
        out.setdefault((qid, key), {})[grader] = entry
    return out


def test_pips_are_positional_and_follow_pool_order() -> None:
    """pips[i] is candidate i's grade — not the i-th grade in the file."""
    q = _query("pt-01", [("film_a", 7), ("film_b", 2), ("film_a", 3)])
    # Deliberately recorded out of pool order.
    per = _per_annotator(
        [
            ("pt-01", grade_scene_key("film_a", 3), "mo", Grade.HIGHLY_RELEVANT),
            ("pt-01", grade_scene_key("film_a", 7), "mo", Grade.IRRELEVANT),
        ]
    )
    (row,) = build_queue_rows([q], per, grader_name="mo")
    assert row.pips == [0, None, 3]
    assert (row.judged, row.total) == (2, 3)
    assert not row.done


def test_judgments_against_a_stale_pool_do_not_complete_a_row() -> None:
    """The regression: a regenerated pool left rows reading done at 56/22.

    The grader has judged more scenes on this query than the pool now
    carries, but only one of them is still a candidate.
    """
    q = _query("pt-02", [("film_a", 1), ("film_a", 2)])
    per = _per_annotator(
        [
            ("pt-02", grade_scene_key("film_a", 1), "mo", Grade.RELEVANT),
            # Three judgments from a pool this run has since replaced.
            ("pt-02", grade_scene_key("film_z", 91), "mo", Grade.RELEVANT),
            ("pt-02", grade_scene_key("film_z", 92), "mo", Grade.WEAKLY),
            ("pt-02", grade_scene_key("film_z", 93), "mo", Grade.IRRELEVANT),
        ]
    )
    (row,) = build_queue_rows([q], per, grader_name="mo")
    assert (row.judged, row.total) == (1, 2)
    assert row.status == "pending"
    assert not row.done


def test_another_graders_judgments_do_not_count_for_this_grader() -> None:
    q = _query("pt-03", [("film_a", 1)])
    per = _per_annotator([("pt-03", grade_scene_key("film_a", 1), "other", Grade.RELEVANT)])
    (row,) = build_queue_rows([q], per, grader_name="mo")
    assert row.pips == [None]
    assert not row.done


def test_skip_counts_as_judged_and_keeps_its_own_pip() -> None:
    """SKIP is a recorded 'no opinion' — judged, and distinct from ungraded."""
    q = _query("pt-04", [("film_a", 1)])
    per = _per_annotator([("pt-04", grade_scene_key("film_a", 1), "mo", Grade.SKIP)])
    (row,) = build_queue_rows([q], per, grader_name="mo")
    assert row.pips == [-1]
    assert (row.judged, row.total) == (1, 1)
    assert row.done


def test_full_judgment_marks_done_and_conflict_comes_from_the_run() -> None:
    q = _query("pt-05", [("film_a", 1), ("film_a", 2)])
    per = _per_annotator(
        [
            ("pt-05", grade_scene_key("film_a", 1), "mo", Grade.RELEVANT),
            ("pt-05", grade_scene_key("film_a", 2), "mo", Grade.WEAKLY),
        ]
    )
    (row,) = build_queue_rows([q], per, grader_name="mo", conflict_ids={"pt-05"})
    assert row.done and row.status == "done"
    assert row.conflict


def test_candidateless_query_is_finished_by_any_judgment() -> None:
    """Mirrors first_ungraded's escape so a malformed record cannot trap the cursor."""
    q = {"id": "pt-06", "results": []}
    assert not build_queue_rows([q], {}, grader_name="mo")[0].done
    per = _per_annotator([("pt-06", "film_a/1", "mo", Grade.RELEVANT)])
    assert build_queue_rows([q], per, grader_name="mo")[0].done


def test_search_text_carries_id_text_lang_and_source_lowercased() -> None:
    q = _query("pt-07", [("film_a", 1)], text="Barco No Mar", lang="pt", source="manual")
    (row,) = build_queue_rows([q], {}, grader_name="mo")
    assert row.search_text == "pt-07 barco no mar pt manual"


def test_numeric_ids_get_the_padded_label() -> None:
    (row,) = build_queue_rows([_query("7", [("film_a", 1)])], {}, grader_name="mo")
    assert row.label == "Q-007"
