"""Pydantic v2 request/response models for the JSON API surface (A3).

Co-located here so A4 (ErrorEnvelope), A5 (HealthStatus/ReadyStatus), and
A7 (Pagination) can all import from a single contract module rather than
scattering models across route files.

All models use Pydantic v2 semantics (model_config, Field factories, etc.).
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field
from pydantic.functional_validators import BeforeValidator


# Empty-string query params sent by HTMX when Alpine :value bindings haven't
# fired yet (Alpine initialises asynchronously after the DOM renders). FastAPI
# validates individual query params before the Pydantic model_validator runs,
# so empty strings must be stripped at the field level via BeforeValidator.
def _empty_str_to_none(v: object) -> object:
    return None if v == "" else v


_OptFloat = Annotated[float | None, BeforeValidator(_empty_str_to_none)]
_OptBool = Annotated[bool | None, BeforeValidator(_empty_str_to_none)]


def _empty_str_top_k(v: object) -> object:
    """Convert empty-string top_k to the field default so HTMX startup quirks don't 422."""
    return 8 if v == "" else v


_TopK = Annotated[int, BeforeValidator(_empty_str_top_k)]

# ---------------------------------------------------------------------------
# Eval — grade + metrics
# ---------------------------------------------------------------------------


class GradeAck(BaseModel):
    """Response body for POST /api/eval/grade."""

    ok: bool
    query_id: str
    scene_id: str
    grade: int


class QueryMetrics(BaseModel):
    """Response body for GET /api/eval/metrics.

    When ``query_id`` is omitted the service returns only ``queries``.
    When ``query_id`` is supplied it returns the per-query metric fields.
    All metric fields are optional so both response shapes validate.
    """

    # Present when query_id is None (list-queries branch)
    queries: list[str] | None = Field(
        default=None,
        description="Graded query IDs available in the active run (query_id=None branch).",
    )
    # Present when query_id is supplied
    p_at_3: float | None = Field(default=None, description="Precision@3")
    p_at_5: float | None = Field(default=None, description="Precision@5")
    ndcg_at_5: float | None = Field(default=None, description="nDCG@5")
    inversions: int | None = Field(default=None, description="Number of grade-order inversions")
    histogram: dict[int, int] | None = Field(
        default=None, description="Grade-value frequency histogram (keyed by grade int value)"
    )


# ---------------------------------------------------------------------------
# Search — typed query params (Depends-able)
# ---------------------------------------------------------------------------


class SearchParams(BaseModel):
    """Typed query parameters for GET /api/search.

    Used via ``Depends(SearchParams)`` so FastAPI surfaces the bounded
    types in /docs (``top_k`` ``ge/le`` → minimum/maximum).

    Note on ``retriever``: it is a plain ``str`` with an OpenAPI ``enum``
    *hint*, NOT a ``Literal``, so it is **not validated** — an unknown value
    is accepted and normalised downstream (``resolve_retriever_args`` warns
    and falls back to ``hybrid``; pinned by
    ``test_search_route_unknown_retriever_falls_back_to_default``). This is
    deliberate: bookmarked/older clients with a stale ``?retriever=`` keep
    working (graceful fallback) instead of getting a 422.
    """

    model_config = {"extra": "ignore"}

    q: str = Field(default="", description="Free-text search query")
    top_k: _TopK = Field(
        default=8,
        ge=1,
        le=200,
        description="Number of results to return (1–200)",
    )
    retriever: str = Field(
        default="hybrid",
        description=(
            "Retrieval backend: clip (semantic), bm25 (keyword), hybrid (RRF fusion). "
            "Accepted-and-normalised: an unknown value falls back to hybrid (not a 422)."
        ),
        json_schema_extra={"enum": ["clip", "bm25", "hybrid"]},
    )
    sem_w: _OptFloat = Field(
        default=None, description="Semantic weight override for hybrid retrieval"
    )
    bm25_w: _OptFloat = Field(default=None, description="BM25 weight override for hybrid retrieval")
    reranker_enabled: _OptBool = Field(
        default=None, description="Override the config default for cross-encoder reranking"
    )


# ---------------------------------------------------------------------------
# System — health / ready (A5)
# ---------------------------------------------------------------------------


class HealthStatus(BaseModel):
    """Response body for GET /api/health."""

    status: Literal["ok"]


class ReadyStatus(BaseModel):
    """Response body for GET /api/ready."""

    ready: bool
    checks: dict[str, bool]


# ---------------------------------------------------------------------------
# Error envelope (A4)
# ---------------------------------------------------------------------------


class ErrorEnvelope(BaseModel):
    """Standard error response body returned by A4 exception handlers."""

    error: str = Field(description="Human-readable error message")
    code: str = Field(description="Machine-readable error code (snake_case)")
    details: dict | None = Field(default=None, description="Optional structured details")
    status: int = Field(description="HTTP status code echoed in the body")


# ---------------------------------------------------------------------------
# Pagination query params (A7)
# ---------------------------------------------------------------------------


class Pagination(BaseModel):
    """Query-parameter pagination model; used via ``Depends(Pagination)`` (A7).

    FastAPI resolves the fields from query-string parameters:
      ``?limit=10&offset=20`` → ``Pagination(limit=10, offset=20)``

    Validation constraints are enforced before the handler runs, so an
    out-of-range ``limit`` (e.g. ``?limit=9999``) returns **422** with
    a structured error body rather than a 500 from the handler.
    """

    model_config = {"extra": "ignore"}

    limit: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Maximum number of items per page (1–200, default 50)",
    )
    offset: int = Field(
        default=0,
        ge=0,
        description="Zero-based offset of the first item (default 0)",
    )
