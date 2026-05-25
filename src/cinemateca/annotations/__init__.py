"""cinemateca.annotations — manual annotation persistence + tag merge."""
from __future__ import annotations

from cinemateca.annotations.io import (
    FILENAME,
    load,
    merge_tag_index,
    save,
)

__all__ = [
    "FILENAME",
    "load",
    "merge_tag_index",
    "save",
]
