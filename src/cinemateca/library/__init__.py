"""cinemateca.library — film registry, scan, per-film context, on-disk utilities.

Public API:

    from cinemateca.library import (
        Film, LibraryState,
        scan_library, library_state,
        register_film, delete_film, load_registry, save_registry,
        # Added in subsequent P2 tasks:
        # FilmContext,                    # T3
        # Library,                        # T10
        # load_json, keyframe_url,        # T4
        # to_smpte, derive_fps,           # T4
        # load_tag_index, load_metadata,  # T5
    )
"""
from __future__ import annotations

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
    "LibraryState",
    "delete_film",
    "library_state",
    "load_registry",
    "register_film",
    "save_registry",
    "scan_library",
]
