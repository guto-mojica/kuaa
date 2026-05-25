"""Data types for the search API. No behavior — just shapes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from cinemateca.retrieval.hybrid import DEFAULT_RRF_K

SearchMode = Literal["clip", "bm25", "hybrid"]


class UploadRejected(Exception):
    """Image upload failed server-side validation (size / content-type / suffix)."""


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
    def text_query(cls, q: str) -> Query:
        return cls(text=q)

    @classmethod
    def image(cls, path: Path) -> Query:
        return cls(image_path=path)


# Compat aliases — the factories read more naturally without the suffix.
# Note: ``Query.text`` is also a dataclass field name; the class-level reassignment
# here only affects attribute access on the class itself (Query.text(...)),
# not on instances (q.text returns the field value).
Query.text = Query.text_query  # type: ignore[method-assign]


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


@dataclass(frozen=True)
class SearchResult:
    """Result of a single search call. ``no_index=True`` carries the empty-state signal."""

    hits: list[Hit]
    mode: SearchMode
    weights: HybridWeights | None
    query: Query
    no_index: bool = False
