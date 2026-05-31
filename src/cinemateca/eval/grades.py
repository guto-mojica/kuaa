"""Eval-set-builder grade persistence (Task 30).

The Eval set builder is the relevance-judgment grading UI used to label
(query, scene) pairs for the v1.0 retrieval eval. Grades are persisted
as an append-only JSONL log per run::

    <eval_root>/<run_id>.jsonl
    {"query_id": "q1", "scene_id": "jeca/1", "grader": "rg",
     "grade": 2, "ts": "2026-05-20T18:42:00+00:00"}

Append-only because re-grading the same (query, scene) pair is normal —
keeping history lets us audit how a grader's opinion evolved and compute
inter-annotator agreement across versions. ``load_run`` resolves the
log by taking the LAST entry for each (query_id, scene_id) key.

Module surface
--------------
``Grade``
    IntEnum with the 5 grade values (IRRELEVANT=0, WEAKLY=1,
    RELEVANT=2, HIGHLY_RELEVANT=3, SKIP=-1). SKIP is a first-class
    "no opinion" sentinel, not the absence of a grade.

``EvalRun``
    Identifier dataclass: ``run_id`` (the eval-set name, e.g.
    ``"month1_curator"``) + ``root`` (a Path that holds the JSONL).

``GradeEntry``
    One row of the JSONL log: query/scene/grader/grade plus the UTC
    timestamp string the writer attached.

``save_grade``
    Append one row. Creates the run root on demand. Caller chooses the
    grader id (the route reads ``grader`` from a cookie).

``load_run``
    Read the JSONL back into a ``LoadedRun`` whose ``grades`` dict
    holds the latest entry per (query_id, scene_id) key.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path


class Grade(IntEnum):
    """Relevance grades for the eval grading UI.

    Numeric values are part of the on-disk JSONL contract:

    * ``0`` IRRELEVANT — definitely not a match
    * ``1`` WEAKLY — tangentially related
    * ``2`` RELEVANT — a reasonable match
    * ``3`` HIGHLY_RELEVANT — exemplary match
    * ``-1`` SKIP — explicit "no opinion" (distinct from "ungraded")
    """

    IRRELEVANT = 0
    WEAKLY = 1
    RELEVANT = 2
    HIGHLY_RELEVANT = 3
    SKIP = -1


@dataclass(frozen=True)
class GradeEntry:
    """One JSONL row."""

    query_id: str
    scene_id: str
    grader: str
    grade: Grade
    ts: str


@dataclass
class EvalRun:
    """Identifier + storage root for a grading run.

    ``run_id`` becomes the JSONL filename stem (``<root>/<run_id>.jsonl``).
    Keep it filesystem-safe — the loader does no escaping.
    """

    run_id: str
    root: Path

    @property
    def jsonl_path(self) -> Path:
        return self.root / f"{self.run_id}.jsonl"


@dataclass
class LoadedRun:
    """In-memory projection of a run JSONL.

    ``grades`` carries one entry per ``(query_id, scene_id)`` key —
    the LATEST entry on file when multiple writes targeted the same
    key (append-only log → last-write-wins reduce).
    """

    run_id: str
    grades: dict[tuple[str, str], GradeEntry] = field(default_factory=dict)


def save_grade(
    run: EvalRun,
    *,
    query_id: str,
    scene_id: str,
    grader: str,
    grade: Grade,
) -> GradeEntry:
    """Append one grade to the run JSONL. Returns the persisted entry.

    Creates ``run.root`` if it does not exist. Each row is one
    self-describing JSON object on its own line; the grade value is
    serialised as an int (the ``Grade`` IntEnum's underlying value)
    so downstream consumers don't need to import ``Grade`` to read.
    """

    run.root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    entry = GradeEntry(
        query_id=query_id,
        scene_id=scene_id,
        grader=grader,
        grade=grade,
        ts=ts,
    )
    payload = {
        "query_id": query_id,
        "scene_id": scene_id,
        "grader": grader,
        "grade": int(grade),
        "ts": ts,
    }
    with run.jsonl_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload) + "\n")
    return entry


def load_run(run: EvalRun) -> LoadedRun:
    """Read a run JSONL into a LoadedRun.

    Missing file → empty ``grades`` dict (a not-yet-started run).
    Re-grades collapse to the LAST entry per ``(query_id, scene_id)``.
    """

    grades: dict[tuple[str, str], GradeEntry] = {}
    if not run.jsonl_path.exists():
        return LoadedRun(run_id=run.run_id, grades=grades)

    for line in run.jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        entry = GradeEntry(
            query_id=record["query_id"],
            scene_id=record["scene_id"],
            grader=record["grader"],
            grade=Grade(int(record["grade"])),
            ts=record["ts"],
        )
        grades[(entry.query_id, entry.scene_id)] = entry

    return LoadedRun(run_id=run.run_id, grades=grades)


def load_run_per_annotator(
    run: EvalRun,
) -> dict[tuple[str, str], dict[str, GradeEntry]]:
    """Read a run JSONL into a per-(query, scene) view that preserves
    each annotator's latest grade.

    The default ``load_run`` collapses across graders — it keeps only
    the LAST GradeEntry per ``(query_id, scene_id)``, so a Rafael→Julia
    re-grade sequence loses Rafael's earlier vote. That collapse is the
    right reduce for per-query metric math (P@K / nDCG operate on one
    canonical grade per scene) but it makes inter-annotator agreement
    measurement impossible — there are no pairs of grades from
    different annotators left to compare.

    This loader keeps the per-grader latest:

        out[(qid, sid)][grader] -> GradeEntry  (latest from that grader)

    Missing file → empty dict (a not-yet-started run). Callers do
    ``shared = set(out[(q, s)]) & {grader_a, grader_b}`` to find the
    overlap two graders rated, which is the basis for κ + the 5×5
    confusion matrix the eval Right Pane renders.
    """

    out: dict[tuple[str, str], dict[str, GradeEntry]] = {}
    if not run.jsonl_path.exists():
        return out

    for line in run.jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        entry = GradeEntry(
            query_id=record["query_id"],
            scene_id=record["scene_id"],
            grader=record["grader"],
            grade=Grade(int(record["grade"])),
            ts=record["ts"],
        )
        key = (entry.query_id, entry.scene_id)
        bucket = out.setdefault(key, {})
        # Last-write-wins per grader — a regrade by the same person
        # supersedes their earlier vote, but does not overwrite the
        # other annotator's vote on the same (q, s).
        bucket[entry.grader] = entry

    return out


def export_run(run: EvalRun) -> dict:
    """Collapse the append-only JSONL to a per-query graded structure.

    Uses :func:`load_run` (last-write-wins reduce) so a re-graded
    ``(query_id, scene_id)`` pair reflects the LATEST grade only.

    Returns::

        {
          "run_id": run.run_id,
          "grades": {query_id: {scene_id: int_grade, ...}, ...},
          "summary": {
              "distinct_pairs": <count of (query_id, scene_id) pairs>,
              "queries": <count of distinct query ids>,
              "graders": <count of distinct annotators>,
          },
        }

    An absent or empty JSONL returns a valid structure with empty grades
    and zero summary counts (a not-yet-started run).
    """
    loaded = load_run(run)

    grades_out: dict[str, dict[str, int]] = {}
    for (qid, sid), entry in loaded.grades.items():
        grades_out.setdefault(str(qid), {})[str(sid)] = int(entry.grade)

    # Count distinct annotators from the raw JSONL (load_run collapses
    # across graders, so we re-read per-annotator just for the grader count).
    per_annot = load_run_per_annotator(run)
    graders: set[str] = set()
    for by_who in per_annot.values():
        graders.update(by_who.keys())

    distinct_pairs = sum(len(scenes) for scenes in grades_out.values())

    return {
        "run_id": run.run_id,
        "grades": grades_out,
        "summary": {
            "distinct_pairs": distinct_pairs,
            "queries": len(grades_out),
            "graders": len(graders),
        },
    }


def grades_by_query(loaded: LoadedRun) -> dict[str, list[Grade]]:
    """Group LoadedRun.grades by query_id."""

    out: dict[str, list[Grade]] = {}
    for (qid, _scene_id), entry in loaded.grades.items():
        out.setdefault(qid, []).append(entry.grade)
    return out


def grades_for_query(loaded: LoadedRun, query_id: str) -> list[Grade]:
    """Return the grade list for a single query (any scene id)."""

    return [entry.grade for (qid, _scene_id), entry in loaded.grades.items() if qid == query_id]


def first_ungraded(
    queries: list[dict],
    per_annotator: dict[tuple[str, str], dict[str, GradeEntry]],
    grader_name: str,
) -> dict | None:
    """Return the first query in ``queries`` that ``grader_name`` has not graded.

    A query is graded by ``grader_name`` when at least one
    ``(query_id, scene_id)`` entry in ``per_annotator`` carries
    ``grader_name`` as a key. Returns ``None`` when the grader has graded
    every query (or ``queries`` is empty) — the /eval context builder falls
    back to ``queries[0]`` so a finished grader doesn't land on None.

    Drives the /eval session-resume landing row (open the page → first
    unjudged query for the active grader). Extracted from the api service so
    the resume rule lives next to the grade-loading primitives it reads.
    """
    # Pre-compute the set of query_ids grader_name has touched to avoid
    # an O(n²) inner scan.
    graded_qids: set[str] = set()
    for (qid, _sid), by_who in per_annotator.items():
        if grader_name in by_who:
            graded_qids.add(str(qid))

    for q in queries:
        qid = str(q.get("id", ""))
        if qid and qid not in graded_qids:
            return q
    return None
