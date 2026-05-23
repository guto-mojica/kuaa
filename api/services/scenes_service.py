"""Scenes service — public re-export surface.

Implementation is split into two sub-modules:
  _scene_detail.py   — single-scene inspector context (build_inspector_context)
  _scenes_list.py    — multi-scene Cenas grid + timeline context
"""

from api.services._scene_detail import (
    _description_for,
    _films_by_slug,
    _resolve_tab,
    _scene_lookup,
    _tags_for,
    build_inspector_context,
    tipo_of,
)
from api.services._scenes_list import (
    _TIPO_DISPLAY_ORDER,
    _TIPO_LABEL,
    _VALID_GROUPS,
    _VALID_SORTS,
    _build_groups_by_film,
    _build_scenes_for_timeline,
    _card_to_scene,
    _compute_timeline_ticks,
    _film_for_grid,
    _format_runtime_hm,
    _format_runtime_tc,
    _hhmm_from_seconds,
    _last_end_time_s,
    _regroup,
    _sort_scenes,
    build_cenas_context,
    build_timeline_context,
)

__all__ = [
    "build_inspector_context",
    "build_cenas_context",
    "build_timeline_context",
    "tipo_of",
    # Internal helpers re-exported for backward compat with any direct callers.
    "_description_for",
    "_films_by_slug",
    "_resolve_tab",
    "_scene_lookup",
    "_tags_for",
    "_TIPO_DISPLAY_ORDER",
    "_TIPO_LABEL",
    "_VALID_GROUPS",
    "_VALID_SORTS",
    "_build_groups_by_film",
    "_build_scenes_for_timeline",
    "_card_to_scene",
    "_compute_timeline_ticks",
    "_film_for_grid",
    "_format_runtime_hm",
    "_format_runtime_tc",
    "_hhmm_from_seconds",
    "_last_end_time_s",
    "_regroup",
    "_sort_scenes",
]
