"""Public search API. See docs/superpowers/specs/2026-05-24-deep-modules-refactor-design.md.

The package exposes a 4-verb / 7-type surface — small, typed, and
behaviour-preserving over P1's M2 hybrid-search baseline. The
under-the-hood modules (``cache``, ``bm25``, ``clip``, ``hybrid``,
``aggregate``, ``_dispatch``, ``_results``, ``_lookup``,
``_tag_index``, ``display``, ``upload``) are private; callers use this
namespace.

Verbs:
  * :func:`find` — single-film search (text or image).
  * :func:`aggregate` — cross-film text search.
  * :func:`reindex_bm25` — flush a film's BM25 cache slot.
  * :func:`rerank` — M2 cross-encoder reranker (stub in P1).

Types:
  * :class:`Query` — query value-object (text / image_path / image_bytes).
  * :class:`Filters` — tag / min-similarity filter.
  * :class:`HybridWeights` — RRF weights + ``rrf_k`` knob.
  * :class:`Hit` — one result row.
  * :class:`SearchResult` — typed return value with ``no_index`` flag.
  * :data:`SearchMode` — ``Literal["clip", "bm25", "hybrid"]``.
  * :class:`UploadRejected` — exception raised by upload validation.
"""

from cinemateca.search._dispatch import find
from cinemateca.search.aggregate import aggregate
from cinemateca.search.bm25 import reindex_bm25
from cinemateca.search.rerank import rerank
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
    # verbs
    "find",
    "aggregate",
    "reindex_bm25",
    "rerank",
    # types
    "Filters",
    "Hit",
    "HybridWeights",
    "Query",
    "SearchMode",
    "SearchResult",
    "UploadRejected",
]
