"""Reranking stub. M2 fills in the cross-encoder model body.

The ``model="noop"`` escape hatch is intentional — it lets a caller
exercise the reranking *plumbing* (DTO mapping, response shape) without
the real cross-encoder model. M2 implementation work removes this stub
file's body but keeps the public signature.
"""
from __future__ import annotations

from cinemateca.search.types import SearchResult


def rerank(result: SearchResult, *, model: str = "default") -> SearchResult:
    """Rerank a search result with a cross-encoder.

    P1 status: stub. ``model="default"`` raises ``NotImplementedError``
    until M2 lands; ``model="noop"`` is a passthrough for callers that
    want to exercise the wiring without the real model.
    """
    if model == "noop":
        return result
    raise NotImplementedError(
        f"M2 cross-encoder reranker not yet implemented; model={model!r}. "
        f"Use model='noop' to passthrough during M2 wiring work."
    )
