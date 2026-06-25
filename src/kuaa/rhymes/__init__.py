"""kuaa.rhymes — cross-film visual similarity (kNN + MMR).

Public API:
    from kuaa.rhymes import Rhyme, find_rhymes, mmr_rerank
    from kuaa.rhymes import description_for, load_scene_meta, resolve_timecode, tags_for
    from kuaa.rhymes import enrich_rhyme, select_echo, shared_tags, signals_for_pair
    from kuaa.rhymes import default_anchor, parse_anchor, rimas_cfg
"""

from __future__ import annotations

from kuaa.rhymes.algorithm import Rhyme, find_rhymes, mmr_rerank
from kuaa.rhymes.anchor import default_anchor, parse_anchor
from kuaa.rhymes.config import rimas_cfg
from kuaa.rhymes.enrich import (
    enrich_rhyme,
    select_echo,
    shared_tags,
    signals_for_pair,
)
from kuaa.rhymes.metadata import (
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
    "mmr_rerank",
    "parse_anchor",
    "resolve_timecode",
    "rimas_cfg",
    "select_echo",
    "shared_tags",
    "signals_for_pair",
    "tags_for",
]
