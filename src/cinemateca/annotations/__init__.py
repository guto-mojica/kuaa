"""cinemateca.annotations — manual annotation persistence + tag merge."""
from __future__ import annotations

from cinemateca.annotations.io import (
    load,
    merge_tag_index,
    save,
)

__all__ = [
    "load",
    "merge_tag_index",
    "save",
]
