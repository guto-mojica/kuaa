"""BM25 loader for a single film's metadata directory.

Caches a :class:`cinemateca.retrieval.bm25.BM25Index` per film, keyed by
the ``(mtime_ns, size)`` stamp of the three on-disk source files
(``scene_descriptions.json``, ``scene_tags.json``,
``manual_annotations.json``). Any write to any of the three invalidates
the entry transparently.

Public verbs:

  * :func:`bm25_index_for_dir` — pure ``Path``-only loader (no app
    config dependency). Used by tests and by ``bm25_index_for_ctx`` once
    the latter has resolved the config tuple.
  * :func:`bm25_index_for_ctx` — context-flavoured loader. Accepts a
    duck-typed ``ctx`` exposing ``metadata_dir`` and the three BM25
    tunables (``stopwords_lang``, ``k1``, ``b``). The thin shim in
    ``api/services/search.py::_get_bm25_index_for_ctx`` reads
    ``cfg.search.bm25`` and forwards.
  * :func:`reindex_bm25` — public M2/P1 verb: drop the BM25 slot for a
    film. ``ctx`` is duck-typed (must expose ``metadata_dir``). Today
    the cache is module-wide, so this clears every slot; a per-slug
    refinement can land in a follow-up without changing this signature.
  * :func:`clear_bm25_cache` — test-isolation hook. Auto-registers with
    :func:`cinemateca.search.cache.register_cache_clearer` at module
    import time so a single ``clear_index_cache()`` call flushes every
    search-cache layer (CLIP + BM25).

The module deliberately does NOT import from ``api.*`` — that boundary
is enforced by import-linter (``cinemateca core must not import api/``).
The merged tag-index composition that used to be inlined here was
promoted to :mod:`cinemateca.search._tag_index` in T8 so the same
loader can be shared with the Mojica context builders without
re-introducing the ``api.services.catalog`` import. Semantics match
byte-for-byte (raw, un-normalised, mixed-key dict; malformed
``scene_tags.json`` tolerated).
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

from cinemateca.retrieval.bm25 import BM25Index
from cinemateca.search._tag_index import load_tag_index
from cinemateca.search.cache import register_cache_clearer

logger = logging.getLogger(__name__)


def _file_stamp(path: Path) -> tuple[int, int]:
    """``(mtime_ns, size)`` of a file, or ``(0, 0)`` if absent.

    Used as a cache key — bumps on any write, including ones that land
    within the same wall-clock second (size differs) AND ones that
    don't change size at all (mtime_ns differs by >=1 ns). Float
    ``st_mtime`` cannot distinguish sub-second writes at modern
    epochs because the IEEE-754 double resolution near 1.7e9 seconds
    is ~2.4e-7 s — quick consecutive writes lose their stamp
    distinction. Nanosecond ints sidestep the problem entirely.
    """
    try:
        st = path.stat()
    except FileNotFoundError:
        return (0, 0)
    return (st.st_mtime_ns, st.st_size)


@lru_cache(maxsize=32)
def _cached_bm25_index(
    metadata_dir: str,
    descriptions_stamp: tuple[int, int],
    scene_tags_stamp: tuple[int, int],
    manual_annotations_stamp: tuple[int, int],
    stopwords_lang: str | None,
    k1: float,
    b: float,
) -> BM25Index:
    """Build a BM25 index for the given (already-stamped) metadata dir.

    The three stamp tuples form the cache key: any write to any of the
    three source files bumps either mtime or size, forcing a rebuild.
    The path string is in the key too so two films don't collide.

    Source files (all under ``metadata_dir``):
      * ``scene_descriptions.json`` — Moondream output (list of dicts).
      * ``scene_tags.json`` — LLM-tag output (INT scene_id keys).
      * ``manual_annotations.json`` — manual tags (STR scene_id keys).

    Tag merge semantics come from :func:`cinemateca.search._tag_index.load_tag_index`
    (the shared loader — single source of truth for scene_id
    normalisation across both tag files within the search package).
    """
    md = Path(metadata_dir)
    descriptions_path = md / "scene_descriptions.json"
    descriptions: list[dict] = []
    if descriptions_path.exists():
        try:
            data = json.loads(descriptions_path.read_text())
            descriptions = data if isinstance(data, list) else []
        except json.JSONDecodeError:
            logger.warning("BM25: malformed %s; using empty descriptions", descriptions_path)

    # Merged LLM ⊕ manual tag index (raw, un-normalised) — the same
    # shape the old api.services.catalog.load_tag_index produced.
    tag_index = load_tag_index(md) or {}

    return BM25Index.build(
        descriptions=descriptions,
        tag_index=tag_index,
        stopwords_lang=stopwords_lang,
        k1=k1,
        b=b,
    )


def bm25_index_for_dir(
    *,
    metadata_dir: Path,
    stopwords_lang: str | None,
    k1: float,
    b: float,
) -> BM25Index:
    """Load (cached) BM25 index for a metadata directory.

    The cache key includes the three file stamps so any write
    invalidates. ``stopwords_lang`` / ``k1`` / ``b`` participate in the
    cache key too, so a config change reloads correctly without a
    manual flush.
    """
    return _cached_bm25_index(
        str(metadata_dir),
        _file_stamp(metadata_dir / "scene_descriptions.json"),
        _file_stamp(metadata_dir / "scene_tags.json"),
        _file_stamp(metadata_dir / "manual_annotations.json"),
        stopwords_lang,
        k1,
        b,
    )


def bm25_index_for_ctx(
    ctx: Any,
    *,
    stopwords_lang: str | None,
    k1: float,
    b: float,
) -> BM25Index:
    """Load BM25 index for a film context (duck-typed).

    ``ctx`` must expose ``metadata_dir``. The three BM25 tunables are
    passed explicitly so this module stays free of any ``api.*``
    import — the call-site shim in ``api/services/search.py`` resolves
    them from ``cfg.search.bm25`` and forwards.
    """
    return bm25_index_for_dir(
        metadata_dir=ctx.metadata_dir,
        stopwords_lang=stopwords_lang,
        k1=k1,
        b=b,
    )


def reindex_bm25(ctx: Any) -> None:
    """Public verb: drop the BM25 cache slot for ``ctx``'s film.

    ``ctx`` is duck-typed (must expose ``metadata_dir``). Today the
    cache is module-wide (clearing one entry clears all because of
    how :func:`functools.lru_cache` works); a per-slug refinement can
    land in a follow-up without changing this signature.
    """
    _cached_bm25_index.cache_clear()


def clear_bm25_cache() -> None:
    """Test-isolation hook.

    Called via the ``cache.register_cache_clearer`` wire below so a
    single ``clear_index_cache()`` call flushes every search-cache
    layer.
    """
    _cached_bm25_index.cache_clear()


# Wire into the central cache clearer so ``clear_index_cache()``
# flushes BM25 too. Module-import-time registration is intentional:
# ``tests/conftest.py`` calls ``clear_index_cache()`` per test, and the
# BM25 lru_cache MUST be flushed alongside the CLIP index cache for
# isolation to hold.
register_cache_clearer(clear_bm25_cache)
