"""cinemateca.rhymes — cross-film visual similarity (kNN + MMR).

Public API:
    from cinemateca.rhymes import Rhyme, find_rhymes
"""
from __future__ import annotations

from cinemateca.rhymes.algorithm import Rhyme, find_rhymes

__all__ = ["Rhyme", "find_rhymes"]
