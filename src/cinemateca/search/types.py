"""Data types for the search API. No behavior — just shapes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from cinemateca.errors import UserInputError
from cinemateca.retrieval.hybrid import DEFAULT_RRF_K

SearchMode = Literal["clip", "bm25", "hybrid"]


class UploadRejected(UserInputError):
    """Image upload failed server-side validation (size / content-type / suffix)."""

    default_code = "input.upload_rejected"


@dataclass(frozen=True)
class Query:
    """A search query. Exactly one of text / image_path / image_bytes must be set."""

    text: str | None = None
    image_path: Path | None = None
    image_bytes: bytes | None = None

    def __post_init__(self) -> None:
        filled = sum(1 for v in (self.text, self.image_path, self.image_bytes) if v is not None)
        if filled != 1:
            raise ValueError("Query must specify exactly one of text / image_path / image_bytes")

    @classmethod
    def of_text(cls, q: str) -> Query:
        """Construct a text query. Mirrors :meth:`image` for symmetry."""
        return cls(text=q)

    @classmethod
    def image(cls, path: Path) -> Query:
        """Construct an image-path query."""
        return cls(image_path=path)


@dataclass(frozen=True)
class HybridWeights:
    """RRF fusion weights. Same defaults as the M2 hybrid search."""

    sem_w: float = 0.70
    bm25_w: float = 0.30
    rrf_k: int = DEFAULT_RRF_K


@dataclass(frozen=True)
class Filters:
    """Tag / similarity filters applied during retrieval."""

    tags: list[str] = field(default_factory=list)
    min_similarity: float = 0.0


@dataclass(frozen=True)
class Hit:
    """One search result row."""

    scene_id: int
    score: float
    keyframe_path: str
    film_slug: str | None = None
    film_title: str | None = None
    timecode: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)
    # Set by :func:`cinemateca.search.rerank.rerank` when the cross-encoder
    # reorders results; ``None`` until reranked (additive, back-compat).
    rerank_score: float | None = None


@dataclass(frozen=True)
class SearchResult:
    """Result of a single search call. ``no_index=True`` carries the empty-state signal."""

    hits: list[Hit]
    mode: SearchMode
    weights: HybridWeights | None
    query: Query
    no_index: bool = False
    # C9 per-query metadata (additive; feeds eval + UI affordances).
    fusion_used: bool = False
    reranker_applied: bool = False
    retriever_mode: str = ""
    num_films_searched: int = 0
    latency_ms: float | None = None

    def __post_init__(self) -> None:
        # Default ``retriever_mode`` to ``mode`` when not explicitly supplied.
        if not self.retriever_mode:
            object.__setattr__(self, "retriever_mode", self.mode)
