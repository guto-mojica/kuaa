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
