"""View model for the /eval queue pane — one row per query.

Every number the queue shows is a statement about **this grader against the
current pool**: which candidates are judged, how many remain, whether the row
is finished. The template used to derive that from
:func:`kuaa.eval.grades.grades_by_query`, which bags every judgment in the
run by query id — across graders, and across pools the run has since
replaced. On ``corpus01`` that read ``56/22`` on one query and marked rows
finished that the resume cursor still had candidates for, so the queue and
the cursor disagreed about the word "done" while sharing a pane.

Building the rows here puts all three surfaces — resume cursor, header
progress, queue row — on :func:`kuaa.eval.grades.judged_grades_by_query` and
:func:`kuaa.eval.grades.candidate_keys`. The template renders what it is
handed and computes nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kuaa.eval.grades import GradeEntry, candidate_keys, judged_grades_by_query

__all__ = ["QueueRow", "build_queue_rows"]


@dataclass(frozen=True)
class QueueRow:
    """One queue entry, fully resolved for rendering."""

    id: str
    label: str
    text: str
    lang: str
    source: str
    #: This grader's grade per candidate, in pool order; ``None`` = unjudged.
    #: Positional, so it is built from :func:`candidate_keys` and nothing else.
    pips: list[int | None] = field(default_factory=list)
    judged: int = 0
    total: int = 0
    done: bool = False
    conflict: bool = False
    #: Lowercased haystack for the queue's search box.
    search_text: str = ""

    @property
    def status(self) -> str:
        """``"done"`` / ``"pending"`` — what the Pending tab filters on."""
        return "done" if self.done else "pending"


def _label(query_id: Any) -> str:
    """``Q-007`` for a numeric id, the id itself for ``pt-01``-style ids."""
    text = str(query_id)
    return f"Q-{int(text):03d}" if text.isdigit() else text


def build_queue_rows(
    queries: list[dict],
    per_annotator: dict[tuple[str, str], dict[str, GradeEntry]],
    *,
    grader_name: str,
    conflict_ids: frozenset[str] | set[str] = frozenset(),
) -> list[QueueRow]:
    """Resolve every query into a :class:`QueueRow` for the queue pane.

    ``conflict_ids`` comes from
    :func:`kuaa.eval.grader_metrics.query_conflict_set` and is a property of
    the run rather than of one grader — it marks queries where two annotators
    disagree by two grades or more, so it stays empty until a run has two.
    """
    judged = judged_grades_by_query(per_annotator, grader_name)
    rows: list[QueueRow] = []
    for q in queries:
        qid = str(q.get("id", ""))
        if not qid:
            continue
        mine = judged.get(qid, {})
        keys = candidate_keys(q)
        pips: list[int | None] = [int(mine[key]) if key in mine else None for key in keys]
        judged_count = sum(1 for pip in pips if pip is not None)
        text = str(q.get("text") or "")
        lang = str(q.get("lang") or "pt")
        source = str(q.get("source") or "manual")
        label = _label(q.get("id"))
        rows.append(
            QueueRow(
                id=qid,
                label=label,
                text=text,
                lang=lang,
                source=source,
                pips=pips,
                judged=judged_count,
                total=len(keys),
                # A query with no candidates cannot be completed by
                # membership, so any judgment finishes it — the same escape
                # ``first_ungraded`` uses to keep a malformed record from
                # trapping the cursor on it forever.
                done=(judged_count == len(keys)) if keys else bool(mine),
                conflict=qid in conflict_ids,
                search_text=f"{label} {text} {lang} {source}".lower(),
            )
        )
    return rows
