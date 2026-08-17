"""Does this pool actually span the retrievers it claims to span?

A pool is only worth grading if each variant in it had a real chance to put
its own candidates in front of the grader. When a variant's config turns out
to be inert, the pool costs a share of generation time, contributes nothing,
and — worse — the ablation table later reports a row whose numbers are another
row's numbers, with no signal that anything went wrong.

That is not hypothetical. Measured on the shipped ``corpus01.queries.json``:
``hybrid_rerank``'s rank map was identical to ``hybrid``'s on 56 of 56
queries. The cross-encoder ran; the slate builder then re-sorted its output by
``Hit.score``, which the reranker does not write, discarding the reordering it
had just paid for. Nothing printed, nothing failed, and the pool advertised
five retrievers while spanning three.

So the report is an artifact next to the pool, and it exits non-zero. A report
nobody is forced to read is how that shipped.

Two checks, and which one applies depends on what kind of retriever it is
(see :attr:`kuaa.eval.registry.RetrieverVariant.derived_from`):

* **Unique contribution** — a *source* retriever (one that reaches the index
  itself: ``clip``, ``bm25``) must propose at least one candidate no other
  variant did. A source that does not is not widening the pool.
* **Distinguishability** — every variant, source or derived, must produce a
  rank map that differs from every other on at least one query. Identical on
  some queries is normal agreement; identical on all of them means one of the
  two is not a distinct retriever.

A *derived* variant — a fusion of other legs, or a reranker over one — is
exempt from the first check and not from the second. Its candidate set is
contained in its parents' by construction, so zero unique candidates is
correct behaviour rather than evidence of anything. On the shipped
``corpus01`` pool ``hybrid`` proposes 560 candidates and 0 unique ones; a
report that failed it for that would be crying wolf next to the one real
finding.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

from kuaa.eval.registry import RETRIEVER_REGISTRY, RetrieverVariant

#: Filename suffix, appended to the run id beside ``<run>.queries.json``.
REPORT_SUFFIX = ".pool_composition.json"


@dataclass
class VariantStats:
    """What one variant contributed across every query in a pool."""

    name: str
    is_source: bool = True
    #: Queries in which the variant proposed at least one candidate.
    queries_contributed: int = 0
    #: Candidates proposed, summed over queries (duplicates across queries count).
    candidates_proposed: int = 0
    #: Candidates no other variant proposed for the same query.
    unique_candidates: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "is_source": self.is_source,
            "queries_contributed": self.queries_contributed,
            "candidates_proposed": self.candidates_proposed,
            "unique_candidates": self.unique_candidates,
        }


@dataclass
class CompositionReport:
    """The composition of a generated pool, plus why it passes or fails."""

    run: str
    query_count: int = 0
    variants: list[VariantStats] = field(default_factory=list)
    #: ``(variant_a, variant_b)`` pairs whose rank maps matched on every query.
    identical_pairs: list[tuple[str, str]] = field(default_factory=list)
    #: Human-readable reasons the pool is not fit to grade. Empty == fit.
    failures: list[str] = field(default_factory=list)
    #: Films seen, sorted — the interleave in ``_merge_across_films`` is
    #: round-robin by sorted slug, so the slug list is part of pool semantics.
    films: list[str] = field(default_factory=list)
    #: Films the generator searched that never reached a pool. Non-empty means
    #: ``k`` fell below the film count and the cut excluded them by name.
    missing_films: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "run": self.run,
            "query_count": self.query_count,
            "films": self.films,
            "missing_films": self.missing_films,
            "variants": [v.to_dict() for v in self.variants],
            "identical_pairs": [list(pair) for pair in self.identical_pairs],
            "failures": self.failures,
            "ok": self.ok,
        }

    def to_text(self) -> str:
        """Render the report for a terminal."""
        lines = [
            f"pool composition — run {self.run!r}: "
            f"{self.query_count} queries over {len(self.films)} film(s)",
            "",
            f"{'variant':<22} {'queries':>8} {'proposed':>9} {'unique':>7}",
        ]
        for v in self.variants:
            suffix = "" if v.is_source else "  (derived)"
            lines.append(
                f"{v.name:<22} {v.queries_contributed:>8} "
                f"{v.candidates_proposed:>9} {v.unique_candidates:>7}{suffix}"
            )
        if self.missing_films:
            lines.append("")
            lines.append(f"never reached any pool: {', '.join(self.missing_films)}")
        if self.identical_pairs:
            lines.append("")
            for a, b in self.identical_pairs:
                lines.append(f"identical rank map on every query: {a} == {b}")
        lines.append("")
        if self.failures:
            lines.append("FAIL — this pool is not fit to grade:")
            lines.extend(f"  - {reason}" for reason in self.failures)
        else:
            lines.append("OK — every variant contributes and no two are indistinguishable.")
        return "\n".join(lines)


def _pool_maps(records: list[dict]) -> list[dict[str, dict[str, int]]]:
    """Per query: ``{variant: {candidate_key: rank}}``.

    ``candidate_key`` is ``"<film_slug>/<scene_id>"`` — the same key
    :func:`kuaa.eval.slates.pool_candidates` dedupes on, so the two views of a
    pool cannot disagree about what a candidate is.
    """
    out: list[dict[str, dict[str, int]]] = []
    for record in records:
        by_variant: dict[str, dict[str, int]] = {}
        for row in record.get("results") or []:
            key = f"{row.get('film_slug', '')}/{row.get('scene_id', '')}"
            for variant, rank in (row.get("pool") or {}).items():
                by_variant.setdefault(str(variant), {})[key] = int(rank)
        out.append(by_variant)
    return out


def analyse_pool(
    records: list[dict],
    *,
    run: str,
    variants: tuple[RetrieverVariant, ...] | None = None,
    expect_text: bool = True,
    library_films: list[str] | None = None,
) -> CompositionReport:
    """Build a :class:`CompositionReport` from generated pool query records.

    ``records`` is what ``kuaa eval slate`` writes to ``<run>.queries.json``.
    Only ``query_type == "text"`` records are examined: image and rhyme
    slates are single-retriever by design (``find`` forces CLIP for image
    queries; rhymes have their own primitive), so there is no span to audit.

    ``variants`` defaults to the registry. Pass a subset to audit a pool that
    was deliberately generated with fewer.

    ``expect_text=False`` for a run generated with ``--modality image`` or
    ``rhyme``: there is nothing to span, so an empty audit is the correct
    outcome rather than a failure.

    ``library_films`` is every slug the generator searched. Films that were
    searched and never surfaced are a failure, because the cross-film merge
    interleaves round-robin by sorted slug: with ``k`` below the film count,
    the cut lands mid-rotation and the last-sorting films are excluded from
    every query in the run, alphabetically rather than by relevance. That is a
    property of ``k`` and the corpus, not of any retriever, so no per-variant
    check can see it — which is exactly why it belongs here rather than in an
    operator's memory of what ``--k`` should be.
    """
    declared = variants if variants is not None else tuple(RETRIEVER_REGISTRY.values())
    text_records = [r for r in records if r.get("query_type") == "text"]
    report = CompositionReport(run=run, query_count=len(text_records))
    stats = {v.name: VariantStats(name=v.name, is_source=v.is_source) for v in declared}

    films: set[str] = set()
    for record in text_records:
        for row in record.get("results") or []:
            films.add(str(row.get("film_slug", "")))
    report.films = sorted(f for f in films if f)

    per_query = _pool_maps(text_records)
    for by_variant in per_query:
        for name, ranks in by_variant.items():
            stat = stats.get(name)
            if stat is None:
                # A pool generated with a variant the registry no longer
                # declares. Surface it rather than dropping it silently — it
                # means the file and the code disagree about the comparison.
                stat = stats[name] = VariantStats(name=name)
            stat.queries_contributed += 1
            stat.candidates_proposed += len(ranks)
        for name, ranks in by_variant.items():
            others: set[str] = set()
            for other_name, other_ranks in by_variant.items():
                if other_name != name:
                    others |= set(other_ranks)
            stats[name].unique_candidates += len(set(ranks) - others)

    report.variants = [stats[v.name] for v in declared] + [
        s for name, s in stats.items() if name not in {v.name for v in declared}
    ]

    # ── failures ────────────────────────────────────────────────────────
    if not text_records:
        if expect_text:
            report.failures.append("no text queries in the pool — nothing to audit")
        return report

    for variant in declared:
        stat = stats[variant.name]
        if stat.queries_contributed == 0:
            report.failures.append(
                f"{variant.name!r} proposed no candidates on any of {len(text_records)} queries"
            )
        elif variant.is_source and stat.unique_candidates == 0:
            report.failures.append(
                f"{variant.name!r} is a source retriever but proposed no candidate "
                f"another variant did not — it costs pool-generation time and "
                f"widens the pool by nothing"
            )

    for a, b in combinations([v.name for v in declared], 2):
        if all(q.get(a) == q.get(b) for q in per_query):
            report.identical_pairs.append((a, b))
            report.failures.append(
                f"{a!r} and {b!r} produced identical rank maps on all "
                f"{len(text_records)} queries — they are not two retrievers"
            )

    unknown = [s.name for s in report.variants if s.name not in {v.name for v in declared}]
    if unknown:
        report.failures.append(
            f"pool carries variant(s) {sorted(unknown)} the registry does not declare"
        )

    if library_films:
        report.missing_films = sorted(set(library_films) - set(report.films))
        if report.missing_films:
            report.failures.append(
                f"film(s) {report.missing_films} were searched and never reached any pool — "
                f"raise --k to at least the film count ({len(set(library_films))}): the "
                f"cross-film merge interleaves round-robin by sorted slug, so a k below "
                f"that excludes the last-sorting films alphabetically, not by relevance"
            )
    return report


def write_report(report: CompositionReport, *, root: Path) -> Path:
    """Write the report beside the pool file. Returns the path written."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{report.run}{REPORT_SUFFIX}"
    path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path


__all__ = [
    "REPORT_SUFFIX",
    "CompositionReport",
    "VariantStats",
    "analyse_pool",
    "write_report",
]
