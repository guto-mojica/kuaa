"""cinemateca.library — film registry, scan, per-film context, on-disk utilities.

Public API:

    from cinemateca.library import (
        Film, LibraryState, FilmContext,
        scan_library, library_state,
        register_film, delete_film, load_registry, save_registry,
        load_json, keyframe_url, to_smpte, derive_fps,
        load_tag_index, load_metadata,
        # Added in subsequent P2 tasks:
        # Library,                        # T10
    )
"""
from __future__ import annotations

from cinemateca.library.context import FilmContext
from cinemateca.library.metadata import (
    load_metadata,
    load_tag_index,
)
from cinemateca.library.paths import (
    derive_fps,
    keyframe_url,
    load_json,
    to_smpte,
)
from cinemateca.library.registry import (
    Film,
    delete_film,
    load_registry,
    register_film,
    save_registry,
)
from cinemateca.library.scan import (
    LibraryState,
    library_state,
    scan_library,
)

__all__ = [
    "Film",
    "FilmContext",
    "LibraryState",
    "delete_film",
    "derive_fps",
    "keyframe_url",
    "library_state",
    "load_json",
    "load_metadata",
    "load_registry",
    "load_tag_index",
    "register_film",
    "save_registry",
    "scan_library",
    "to_smpte",
]
