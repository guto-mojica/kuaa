"""Failure-surfacing for the WS-4 eval harness (E4).

Picks the worst-scoring queries from a multi-retriever text eval and renders a
structured markdown stub per case — the per-retriever first-relevant ranks, the
top *non-relevant* results each carrying the REAL Moondream caption the
retriever saw, and empty ``Hypothesis`` / ``Mitigation`` anchors for a human to
fill from that evidence. The discipline mirrors the existing
``docs/FAILURE_ANALYSIS.md`` M2 cases: every cited rank and every cited caption
is real; nothing is paraphrased or invented.

This module is the pure core. It consumes per-query *records* — already
projected dicts (see :func:`worst_queries`) — and never touches a model or an
index itself. The CLI (``scripts/analyze_failures.py``) runs the real
retrievers, builds the records (enriching the top-wrong rows with the on-disk
Moondream descriptions, since the text path's ``top_results`` carry no
``description`` key), and writes the rendered stubs into the M4 block of the doc.

Layering: core (``cinemateca.*``); MUST NOT import ``api.*`` (import-linter).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Sentinel rendered when a surfaced wrong-result scene has no Moondream caption
# on disk. The hard rule is to never fabricate a description; an honest marker
# is rendered instead so the doc reader knows the gap is real, not an omission.
NO_DESCRIPTION = "(no description on disk)"


@dataclass(frozen=True)
class FailureCase:
    """One worst-scoring query, projected for the failure-analysis doc.

    Attributes:
        query_id: The query's stable id (e.g. ``"text-09"``).
        query_text: The query string, rendered verbatim in the stub header.
        metric_value: The value of the ``by`` metric this case was ranked on
            (lower = worse). Carried so the stub can print it without the caller
            re-deriving which metric drove the selection.
        first_relevant_rank_by_retriever: ``{retriever_name -> rank}`` where
            ``rank`` is the 1-based position of the first relevant scene in that
            retriever's ranking, or ``None`` when no relevant scene was retrieved.
        top_wrong: The top-K NON-relevant results, in rank order. Each is a dict
            carrying at least ``scene_id`` and ``description`` (the real Moondream
            caption the retriever saw; the empty string when none is on disk).
        missing_relevant: Relevant scene ids that never appeared in ANY
            retriever's ranking — the queries' hardest misses.
    """

    query_id: str
    query_text: str
    metric_value: float
    first_relevant_rank_by_retriever: dict[str, int | None]
    top_wrong: tuple[dict[str, Any], ...]
    missing_relevant: tuple[str, ...]

    def to_markdown_stub(self) -> str:
        """Render the case as a ``## <id> — "<text>"`` markdown section.

        The section carries (1) a metrics/ranks block — the ``by`` metric value
        plus the per-retriever first-relevant rank (``None`` shown as an em
        dash, matching the M2 cases' ``"—"`` convention), (2) the top-wrong
        list with each scene's id and its EXACT Moondream caption (or the
        :data:`NO_DESCRIPTION` sentinel when none is on disk), (3) the
        missing-relevant scene ids, and (4) empty ``**Hypothesis:**`` /
        ``**Mitigation:**`` lines for a human to fill from the evidence above.
        """
        lines: list[str] = [f'## {self.query_id} — "{self.query_text}"', ""]

        # Per-retriever first-relevant rank table (em dash = never retrieved).
        retrievers = list(self.first_relevant_rank_by_retriever)
        header = "| | " + " | ".join(retrievers) + " |"
        sep = "| --- | " + " | ".join("---:" for _ in retrievers) + " |"
        rank_cells = [
            (
                "—"
                if self.first_relevant_rank_by_retriever[r] is None
                else str(self.first_relevant_rank_by_retriever[r])
            )
            for r in retrievers
        ]
        rank_row = "| First relevant rank | " + " | ".join(rank_cells) + " |"
        lines += [header, sep, rank_row, ""]

        lines.append(f"Worst-metric value (`{self._metric_label()}`): **{self.metric_value:.3f}**.")
        lines.append("")

        # Top wrong results — id + verbatim Moondream caption (the BM25/CLIP saw).
        lines.append(
            "**Top non-relevant results (rank order) — Moondream caption each retriever saw:**"
        )
        lines.append("")
        if self.top_wrong:
            for i, row in enumerate(self.top_wrong, start=1):
                sid = row.get("scene_id", "?")
                desc = str(row.get("description") or "").strip()
                shown = desc if desc else NO_DESCRIPTION
                lines.append(f"{i}. scene **{sid}** — *{shown}*")
        else:
            lines.append("_(none — every top-K result was a relevant scene)_")
        lines.append("")

        # Missing relevant scenes (never retrieved by any mode).
        if self.missing_relevant:
            lines.append(
                "**Relevant scenes never retrieved:** "
                + ", ".join(str(s) for s in self.missing_relevant)
                + "."
            )
            lines.append("")

        # Empty prose anchors for the human pass (filled from the evidence above).
        lines += ["**Hypothesis:** ", "", "**Mitigation:** ", ""]
        return "\n".join(lines)

    def _metric_label(self) -> str:
        """Best-effort label for the ranked metric; overridden via attribute set."""
        return getattr(self, "_by", "ndcg_at_10")


def worst_queries(
    records: list[dict[str, Any]],
    *,
    n: int = 8,
    by: str = "ndcg_at_10",
) -> list[FailureCase]:
    """Return the ``n`` worst per-query records as :class:`FailureCase`s.

    ``records`` is per-query data the caller has already assembled — one dict
    per query with this shape::

        {
            "query_id": str,
            "query_text": str,
            "metrics": {<metric-name>: float, ...},   # MUST contain ``by``
            "first_relevant_rank_by_retriever": {str: int | None},
            "top_wrong": tuple[dict, ...],            # non-relevant rows
            "missing_relevant": tuple[str, ...],
        }

    The ``by`` metric (default ``"ndcg_at_10"``) is read from each record's
    ``metrics`` block; records are sorted ASCENDING by it (worst first) and the
    first ``n`` are projected into :class:`FailureCase`s. ``n`` larger than the
    corpus returns every record. A stable secondary sort on ``query_id`` keeps
    ties deterministic across runs (so the doc diff is reproducible).

    The returned cases carry the ``by`` label so :meth:`FailureCase.to_markdown_stub`
    can print which metric drove the ranking.
    """

    def _key(rec: dict[str, Any]) -> tuple[float, str]:
        metrics = rec.get("metrics") or {}
        return float(metrics.get(by, 0.0)), str(rec.get("query_id", ""))

    ordered = sorted(records, key=_key)
    out: list[FailureCase] = []
    for rec in ordered[: max(n, 0)]:
        metrics = rec.get("metrics") or {}
        case = FailureCase(
            query_id=str(rec["query_id"]),
            query_text=str(rec.get("query_text", "")),
            metric_value=float(metrics.get(by, 0.0)),
            first_relevant_rank_by_retriever=dict(
                rec.get("first_relevant_rank_by_retriever") or {}
            ),
            top_wrong=tuple(rec.get("top_wrong") or ()),
            missing_relevant=tuple(rec.get("missing_relevant") or ()),
        )
        # Stash the ranked-metric label on the frozen instance so the stub can
        # print it. ``object.__setattr__`` is the documented escape hatch for a
        # frozen dataclass; this private attr is not part of the public field set.
        object.__setattr__(case, "_by", by)
        out.append(case)
    return out


__all__ = ["FailureCase", "NO_DESCRIPTION", "worst_queries"]
