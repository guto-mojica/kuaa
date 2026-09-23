"""One name-space for the retrievers under comparison.

Every consumer of "which retrievers are we comparing?" reads this module:
the pool (:mod:`kuaa.eval.slates`), the ablation table
(:mod:`kuaa.eval.ablation`), and the pool composition report
(:mod:`kuaa.eval.composition`).

Why one registry rather than a list per consumer: grades persist the
*pool* spelling of a variant name (``clip`` / ``hybrid_no_metadata`` /
``hybrid_rerank``). When the ablation table spells the same five rows
differently, a graded pool cannot be joined to the table that is supposed to
score it, and nothing raises — the join simply finds no rows. Deriving one
name-space from the other closes the drift that exists today and reopens it
on the next variant; a single definition does not.

The registry is ordered. Pool generation, the composition report and the
ablation table all iterate it in declaration order, so a pool file's
``pool_variants`` list and a table's row order agree by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kuaa.search import SearchMode


@dataclass(frozen=True)
class RetrieverVariant:
    """One retriever configuration that contributes candidates to a pool.

    ``cfg_overrides`` are applied to ``cfg.search`` for this call only — the
    metadata leg has no ``find`` parameter, it is resolved from config by
    :func:`kuaa.retrieval.hybrid.resolve_metadata_w`, so turning it off means
    handing ``find`` a config that says so.

    ``derived_from`` names the variants whose candidates this one can only
    re-rank or fuse, never extend. Two kinds qualify and both matter to the
    composition report:

    * a **fusion** — ``hybrid`` blends the CLIP and BM25 lists, so its
      candidate set is contained in their union by construction;
    * a **reranker** — the cross-encoder scores the hits it is handed.

    Either *may* contribute no candidate the variants it derives from did not
    already propose, so the report's unique-contribution check cannot be
    applied to it: failing a variant for that would fail it on its definition
    rather than on its behaviour. What a derived variant must instead show is
    a rank map distinguishable from the variants it derives from — that is
    what makes its leg separately scorable after grading.

    Whether a derived variant widens the pool is therefore a measurement, not
    a property. On the shipped ``corpus01`` pool (see
    ``data/eval/corpus01.pool_composition.json``) both do: ``hybrid`` proposes
    15 candidates unique to it of 560, and ``hybrid_rerank`` 159 of 560,
    because a first stage widened to the reranker's input window reaches rows
    the un-widened legs truncate away.

    An empty ``derived_from`` marks a **source**: a retriever that reaches the
    index on its own. A source that proposes nothing unique is doing nothing.
    """

    name: str
    mode: SearchMode = "clip"
    rerank: bool = False
    cfg_overrides: dict[str, Any] = field(default_factory=dict)
    derived_from: tuple[str, ...] = ()
    #: Rendered verbatim as the ablation table's per-row footnote. Empty
    #: means the row needs no explanation beyond its name.
    footnote: str = ""

    @property
    def is_source(self) -> bool:
        """True when this variant reaches the index itself rather than re-ranking."""
        return not self.derived_from

    @property
    def metadata_w(self) -> float | None:
        """The metadata leg's fusion share, or ``None`` to use the config default."""
        raw = self.cfg_overrides.get("hybrid_metadata_w")
        return None if raw is None else float(raw)


#: The retrievers under comparison, in declaration order.
#:
#: ``hybrid_rerank`` earns its place twice over: its rank map is what makes
#: the reranker decision (``docs/RERANKER_DECISION.md``) answerable from
#: grades instead of from judgment, and on ``corpus01`` it also widens the
#: pool by 159 candidates no other variant proposed. Both are checks it has
#: to keep passing — a rank map distinguishable from ``hybrid``'s is what the
#: composition report requires of a derived variant, and a variant that stops
#: clearing it is one the reranker's config has turned inert.
RETRIEVER_REGISTRY: dict[str, RetrieverVariant] = {
    v.name: v
    for v in (
        RetrieverVariant(name="clip"),
        RetrieverVariant(name="bm25", mode="bm25"),
        RetrieverVariant(name="hybrid", mode="hybrid", derived_from=("clip", "bm25")),
        RetrieverVariant(
            name="hybrid_no_metadata",
            mode="hybrid",
            cfg_overrides={"hybrid_metadata_w": 0.0},
            derived_from=("clip", "bm25"),
            footnote=(
                "Identical to `hybrid` except the exact-lexical metadata leg "
                "(tags / descriptions / detected objects) is disabled "
                "(`metadata_w=0`) — the delta to the `hybrid` row isolates that "
                "signal's contribution."
            ),
        ),
        RetrieverVariant(
            name="hybrid_rerank",
            mode="hybrid",
            rerank=True,
            derived_from=("hybrid",),
            footnote=(
                "The bge-reranker-v2-m3 cross-encoder applied on top of the "
                "`hybrid` leg. It reads a first stage widened to the reranker's "
                "input window (`retrieval.reranker.top_k_in`), reorders it, and "
                "cuts back to `k` — so it can promote a candidate `hybrid` "
                "truncated away, and it does add rows to the pool. Compare it to "
                "the `hybrid` row it sits on, not to `clip`."
            ),
        ),
    )
}

#: Stock CLIP, no overrides. The image path uses this directly: ``find``
#: forces CLIP for image queries regardless of ``mode``, so pooling an image
#: query across text retrievers would call five variants to get one answer.
CLIP_ONLY: RetrieverVariant = RETRIEVER_REGISTRY["clip"]

#: The retrievers a text pool spans — the same set the ablation table reports,
#: so every row it compares had a chance to propose candidates.
POOL_VARIANTS: tuple[RetrieverVariant, ...] = tuple(RETRIEVER_REGISTRY.values())


def variant_names() -> tuple[str, ...]:
    """Registry names in declaration order."""
    return tuple(RETRIEVER_REGISTRY)


__all__ = [
    "CLIP_ONLY",
    "POOL_VARIANTS",
    "RETRIEVER_REGISTRY",
    "RetrieverVariant",
    "variant_names",
]
