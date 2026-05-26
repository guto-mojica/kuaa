"""cinemateca.annotations — manual annotation persistence + tag merge."""

from __future__ import annotations

from cinemateca.annotations.descriptions import save_description
from cinemateca.annotations.io import (
    FILENAME,
    load,
    load_annotations,
    merge_tag_index,
    normalize_tags,
    save,
    save_annotations,
)
from cinemateca.annotations.scenes import (
    build_scene_list,
    resolve_selected_film,
    scene_context,
    scene_list_with_fallback,
)

__all__ = [
    "FILENAME",
    "build_scene_list",
    "load",
    "load_annotations",
    "merge_tag_index",
    "normalize_tags",
    "resolve_selected_film",
    "save",
    "save_annotations",
    "save_description",
    "scene_context",
    "scene_list_with_fallback",
]
