"""A10: typed template-context contracts (TypedDict per page/partial).

One TypedDict per major page/partial.  These mirror the EXACT keys each
``build_*`` function returns today — transcribed from the real builder
source, not inferred from templates.

Annotate builder signatures with these types so mypy can catch key-shape
mismatches at the boundary between service and template layers.

``total=False`` is used where ALL keys may be absent (e.g. when the
return value is ``dict | None``).  For builders that always return every
key, ``total=True`` (the default) keeps mypy's completeness checking
fully effective.
"""

from __future__ import annotations

from typing import Any, TypedDict

# ---------------------------------------------------------------------------
# CenasContext
# Builder: api.services.scenes._cards.build_cenas_context
# ---------------------------------------------------------------------------


class _CenasContextRequired(TypedDict):
    """Required keys for the Cenas grid context — always returned by the builder."""

    groups_by_film: list[dict[str, Any]]
    selected_scene_id: int | None
    # UNPAGED totals — shown in the countrow, not affected by pagination.
    total_scenes: int
    film_count: int
    total_runtime_s: float
    total_runtime_str: str
    total_keyframes_size: str
    # Toolrow pip counts.
    visible_field_count: int
    active_filter_count: int
    no_data: bool
    available_tags: list[str]
    # Legacy flat-list retained for backward-compat while Phase 3 cleanup lands.
    cards: list[dict[str, Any]]
    # Active filter state echoed back for full-page navigation preservation.
    query: str
    active_group: str
    active_sort: str
    active_bucket: str
    # Pagination signals consumed by the grid partial's nav bar.
    has_more: bool
    has_prev: bool
    current_offset: int
    next_offset: int
    prev_offset: int
    current_limit: int


class CenasContext(_CenasContextRequired, total=False):
    """Template context for the Cenas grid (``partials/scenes.html``).

    Optional sentinel keys are set by the route (not the builder) to seed
    Alpine reactive state for infinite-scroll / filter preservation.
    """

    sentinel_sort: str
    sentinel_group: str
    sentinel_bucket: str
    sentinel_q: str
    sentinel_tags: list[str]
    sentinel_slug: str


# ---------------------------------------------------------------------------
# SearchContext
# Builder: kuaa.search._lookup.build_search_context
#          kuaa.search._lookup.build_search_context_aggregate
#          (both merge mojica_search_defaults() + available_tags + films_by_id)
# ---------------------------------------------------------------------------


class SearchContext(TypedDict):
    """Template context for the Buscar tab (``partials/search.html``)."""

    query: str
    total: int
    film_count: int
    latency_ms: float | None
    active_mode: str
    active_view: str
    selected_scene_id: int | None
    results: list[dict[str, Any]]
    films_by_id: dict[str, Any]
    highlighted_tags: set[str]
    available_tags: list[str]


# ---------------------------------------------------------------------------
# InspectorContext
# Builder: api.services.scenes._inspector.build_inspector_context
#          Returns ``dict | None``; None on unresolvable (slug, scene_id) pair.
# ---------------------------------------------------------------------------


class InspectorContext(TypedDict, total=False):
    """Template context for the right-pane inspector (``partials/*_inspector.html``).

    ``total=False`` because ``inspector_kind`` is added by the route after
    calling the builder (it is resolved from the HTTP ``kind`` query param,
    not from the data layer).  All other fields are always present when the
    builder returns a non-None value.
    """

    selected_scene: dict[str, Any]
    selected_film: Any  # kuaa.library.Film or None when unregistered
    inspector_tab: str
    rhymes: list[dict[str, Any]]
    inspector_kind: str


# ---------------------------------------------------------------------------
# TimelineContext
# Builder: api.services.scenes._timeline.build_timeline_context
#          Returns ``dict | None``; None when slug/scene_id don't resolve.
# ---------------------------------------------------------------------------


class TimelineContext(TypedDict):
    """Template context for the bottom timeline (``partials/search.html .b-tl``)."""

    selected_film: Any  # types.SimpleNamespace wrapping Film + timeline attrs
    selected_scene: dict[str, Any]
    film_match_n: int
    query: str


# ---------------------------------------------------------------------------
# ProcessingContext
# Builder: api.services.processing_render.build_processing_context
# ---------------------------------------------------------------------------


class ProcessingContext(TypedDict):
    """Template context for the Processing tab (``partials/processing.html``)."""

    films: list[Any]  # list[kuaa.library.Film]
    step_defs: Any  # STEP_DEFS constant from api.jobs
    jobs: list[Any]  # list of enriched JobState
    initial_log_lines: list[dict[str, Any]]
    stats: Any  # aggregate stats namespace from processing_service
    job_queue: list[Any]  # recent-job history for .p-queue
    active_step: Any | None  # sub-step detail; None when no job is running
    gpu_metrics: list[Any]  # CPU/RAM/VRAM metrics; empty list when disabled
    cfg: Any  # the merged effective config namespace


# ---------------------------------------------------------------------------
# RimasContext
# Builder: api.services.rhymes_service.build_rimas_context
# ---------------------------------------------------------------------------


class RimasContext(TypedDict):
    """Template context for the Rimas Visuais tab (``partials/rimas.html``)."""

    anchor_film: Any | None  # kuaa.library.Film or None
    anchor_scene: Any | None  # scene metadata dict or None (empty-state trigger)
    echoes: list[dict[str, Any]]
    selected_echo: dict[str, Any] | None
    selected_echo_id: int | None
    shared_tags: list[str]
    k: int
    mmr_lambda: float
    k_candidates: int
    threshold: float
    library_has_scenes: bool
