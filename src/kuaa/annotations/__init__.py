"""kuaa.annotations — manual annotation persistence + tag merge."""

from __future__ import annotations

from kuaa.annotations.descriptions import save_description
from kuaa.annotations.io import (
    FILENAME,
    load,
    load_annotations,
    merge_tag_index,
    normalize_tags,
    save,
    save_annotations,
)
from kuaa.annotations.overrides import (
    OVERRIDES_FILENAME,
    load_overrides,
    save_overrides,
    set_suppressed,
    suppressed_for_scene,
)
from kuaa.annotations.scenes import (
    build_scene_list,
    resolve_selected_film,
    scene_context,
    scene_list_with_fallback,
)

__all__ = [
    "FILENAME",
    "OVERRIDES_FILENAME",
    "build_scene_list",
    "load",
    "load_annotations",
    "load_overrides",
    "merge_tag_index",
    "normalize_tags",
    "resolve_selected_film",
    "save",
    "save_annotations",
    "save_description",
    "save_overrides",
    "scene_context",
    "scene_list_with_fallback",
    "set_suppressed",
    "suppressed_for_scene",
]
