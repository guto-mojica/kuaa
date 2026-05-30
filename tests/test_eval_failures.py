"""Hermetic unit tests for the failure-surfacing tool (E4).

These cover the pure logic of ``cinemateca.eval.failures`` — the
``worst_queries`` selector (ordering + FailureCase construction) and the
``FailureCase.to_markdown_stub`` renderer (real-description citation +
metrics/ranks block) — WITHOUT loading any model or real index. The GPU
end-to-end acceptance (``scripts/analyze_failures.py`` over the live SigLIP2
library) is run by hand and its output pasted into ``docs/FAILURE_ANALYSIS.md``;
this file is the CI-without-index half.

A per-query *record* is the structure ``worst_queries`` consumes — a dict the
CLI builds by grouping each query's ``RetrievalResult`` across the three text
retrievers. The shape is documented on ``worst_queries`` and exercised here
with hand-built fakes (no ``RetrievalRun`` needed).
"""

from __future__ import annotations

from cinemateca.eval.failures import FailureCase, worst_queries


def _record(
    qid: str,
    *,
    ndcg: float,
    text: str = "a query",
    ranks: dict[str, int | None] | None = None,
    top_wrong: tuple[dict, ...] = (),
    missing: tuple[str, ...] = (),
) -> dict:
    """Build one per-query record in the shape ``worst_queries`` consumes.

    The ``by`` metric defaults to ndcg_at_10; ``ranks`` is the
    retriever→first-relevant-rank map; ``top_wrong`` are the non-relevant rows.
    """
    return {
        "query_id": qid,
        "query_text": text,
        "metrics": {"ndcg_at_10": ndcg, "recall_at_5": 0.0, "recall_at_10": 0.0},
        "first_relevant_rank_by_retriever": ranks or {"clip": None, "bm25": None, "hybrid": None},
        "top_wrong": top_wrong,
        "missing_relevant": missing,
    }


def test_worst_queries_orders_by_metric():
    """worst_queries returns the n lowest-by-metric records, worst (lowest) first,
    each as a FailureCase carrying the query id/text/ranks/top_wrong."""
    records = [
        _record("q1", ndcg=0.90, text="best one"),
        _record("q2", ndcg=0.10, text="worst one", ranks={"clip": 7, "bm25": None, "hybrid": 5}),
        _record("q3", ndcg=0.50, text="middling"),
        _record("q4", ndcg=0.25, text="second worst", top_wrong=({"scene_id": "9", "description": "x"},)),
        _record("q5", ndcg=0.40, text="third worst"),
    ]

    cases = worst_queries(records, n=3, by="ndcg_at_10")

    # Three returned, ascending by metric (worst first): q2(0.10) q4(0.25) q5(0.40).
    assert [c.query_id for c in cases] == ["q2", "q4", "q5"]
    assert [c.metric_value for c in cases] == [0.10, 0.25, 0.40]

    # Each FailureCase carries the projected fields.
    worst = cases[0]
    assert isinstance(worst, FailureCase)
    assert worst.query_id == "q2"
    assert worst.query_text == "worst one"
    assert worst.first_relevant_rank_by_retriever == {"clip": 7, "bm25": None, "hybrid": 5}
    assert worst.top_wrong == ()

    # top_wrong is carried through when present on the record.
    second = cases[1]
    assert second.query_id == "q4"
    assert second.top_wrong == ({"scene_id": "9", "description": "x"},)


def test_worst_queries_respects_n_and_default():
    """n caps the result; the default by-metric is ndcg_at_10."""
    records = [_record(f"q{i}", ndcg=i / 10) for i in range(10)]
    # Default n=8.
    assert len(worst_queries(records)) == 8
    # n larger than the corpus returns all, still sorted ascending.
    everything = worst_queries(records, n=99)
    assert len(everything) == 10
    assert [c.metric_value for c in everything] == [i / 10 for i in range(10)]


def test_stub_includes_real_description():
    """A FailureCase whose top_wrong row carries a Moondream description renders
    that EXACT string in to_markdown_stub() — the tool must cite the real caption
    the retriever saw, never paraphrase it."""
    caption = 'A man in a hat and coat smokes a pipe in a dimly lit room.'
    case = FailureCase(
        query_id="text-09",
        query_text="homem com chapéu fumando dentro de casa",
        metric_value=0.123,
        first_relevant_rank_by_retriever={"clip": 4, "bm25": None, "hybrid": 6},
        top_wrong=(
            {"scene_id": "223", "description": caption},
            {"scene_id": "7", "description": "Another unrelated caption."},
        ),
        missing_relevant=("36",),
    )

    stub = case.to_markdown_stub()

    # Header with id + verbatim query text.
    assert '## text-09 — "homem com chapéu fumando dentro de casa"' in stub
    # The EXACT Moondream caption is present (no truncation, no paraphrase).
    assert caption in stub
    # The wrong scene id is cited next to its caption.
    assert "223" in stub
    # The per-retriever first-relevant ranks are rendered (None → an em dash).
    assert "clip" in stub.lower()
    assert "—" in stub  # bm25 had no relevant scene in top-k → em dash
    # Missing-relevant scene ids surfaced.
    assert "36" in stub
    # Empty prose anchors for the human to fill.
    assert "**Hypothesis:**" in stub
    assert "**Mitigation:**" in stub


def test_stub_handles_missing_description_honestly():
    """When a wrong row has no description, the stub says so rather than inventing
    one (hard rule: never fabricate a caption)."""
    case = FailureCase(
        query_id="text-03",
        query_text="festa popular",
        metric_value=0.0,
        first_relevant_rank_by_retriever={"clip": None, "bm25": None, "hybrid": None},
        top_wrong=({"scene_id": "275", "description": ""},),
        missing_relevant=(),
    )
    stub = case.to_markdown_stub()
    assert "275" in stub
    # A documented sentinel, not a fabricated caption.
    assert "(no description on disk)" in stub
