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

__all__ = [
    "FILENAME",
    "load",
    "load_annotations",
    "merge_tag_index",
    "normalize_tags",
    "save",
    "save_annotations",
    "save_description",
]
