"""cinemateca.rhymes — cross-film visual similarity (kNN + MMR).

Public API:
    from cinemateca.rhymes import Rhyme, find_rhymes
    from cinemateca.rhymes import description_for, load_scene_meta, resolve_timecode, tags_for
"""
from __future__ import annotations

from cinemateca.rhymes.algorithm import Rhyme, find_rhymes
from cinemateca.rhymes.metadata import (
    description_for,
    load_scene_meta,
    resolve_timecode,
    tags_for,
)

__all__ = [
    "Rhyme",
    "description_for",
    "find_rhymes",
    "load_scene_meta",
    "resolve_timecode",
    "tags_for",
]
