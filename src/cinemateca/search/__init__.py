"""Public search API. See docs/superpowers/specs/2026-05-24-deep-modules-refactor-design.md."""

from cinemateca.search.types import (
    Filters,
    Hit,
    HybridWeights,
    Query,
    SearchMode,
    SearchResult,
    UploadRejected,
)

__all__ = [
    "Filters",
    "Hit",
    "HybridWeights",
    "Query",
    "SearchMode",
    "SearchResult",
    "UploadRejected",
]
