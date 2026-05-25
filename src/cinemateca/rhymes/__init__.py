"""cinemateca.rhymes — cross-film visual similarity (kNN + MMR).

Public API:
    from cinemateca.rhymes import Rhyme, find_rhymes
    from cinemateca.rhymes import description_for, load_scene_meta, resolve_timecode, tags_for
    from cinemateca.rhymes import enrich_rhyme, select_echo, shared_tags, signals_for_pair
    from cinemateca.rhymes import default_anchor, parse_anchor, rimas_cfg
"""
from __future__ import annotations

from cinemateca.rhymes.algorithm import Rhyme, find_rhymes
from cinemateca.rhymes.anchor import default_anchor, parse_anchor
from cinemateca.rhymes.config import rimas_cfg
from cinemateca.rhymes.enrich import (
    enrich_rhyme,
    select_echo,
    shared_tags,
    signals_for_pair,
)
from cinemateca.rhymes.metadata import (
    description_for,
    load_scene_meta,
    resolve_timecode,
    tags_for,
)

__all__ = [
    "Rhyme",
    "default_anchor",
    "description_for",
    "enrich_rhyme",
    "find_rhymes",
    "load_scene_meta",
    "parse_anchor",
    "resolve_timecode",
    "rimas_cfg",
    "select_echo",
    "shared_tags",
    "signals_for_pair",
    "tags_for",
]
