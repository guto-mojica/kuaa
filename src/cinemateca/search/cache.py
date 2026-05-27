"""CLIP search-index loader with mtime/size-keyed cache invalidation.

Returns a typed :class:`SearchIndex` instead of raising; callers render
the no-index UI state for ``MISSING`` / ``CORRUPT`` results.

This module is the canonical home for the loader + cache. The legacy
import path ``api.services.search.{IndexStatus, SearchIndex, load_index,
clear_index_cache}`` continues to work via re-exports for back-compat
with existing tests; new code should import from here directly.

Design notes inherited verbatim from the previous home in
``api.services.search`` (do not re-document at the call site):

  * **mtime/size-aware invalidation.** The cache key is
    ``(slug, embeddings_path, mapping_path)`` and the cache value is
    keyed-by-signature on ``(_stat_sig(emb), _stat_sig(map))``. A
    regenerated index (different mtime or size) is reloaded
    transparently. Acknowledged blind spot: a byte-identical regen with
    identical ``st_mtime_ns`` would slip past — practically impossible
    on a real write, and a content hash is the only complete fix
    (deliberately not done for a single-worker dev server).
  * **Index shape validation.** ``OpenClipEmbedder.load`` performs NO
    row-count consistency check; validation lives here so the AI core's
    contract stays untouched and we degrade to a clear "corrupt index"
    UI state rather than crashing ``/api/search`` with an ``IndexError``.
  * **Thread-safety.** A single ``threading.Lock`` covers the
    stat-check + (re)load + store. The lock is held across the disk
    load, which is acceptable for the single-worker dev server.

Duck-typed ``ctx``: ``load_index`` accepts any object exposing
``.slug`` (``str | None``) and ``.embeddings_dir`` (``Path``). The
current producer is :class:`api.services.film_context.FilmContext`; P2
plans to swap in ``cinemateca.library.Library`` without changing this
surface — that is why ``FilmContext`` is deliberately NOT imported
here.

Sibling cache hook: :func:`register_cache_clearer` lets a sibling
module (e.g. ``cinemateca.search.bm25``) plug its own flusher into
:func:`clear_index_cache`, so the test-isolation contract — "one call
flushes every search-cache layer" — survives the package split.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class IndexStatus(str, Enum):
    """Outcome of attempting to load the search index."""

    OK = "ok"
    MISSING = "missing"  # .npy and/or mapping file absent
    CORRUPT = "corrupt"  # files present but shape-inconsistent / unreadable


@dataclass(frozen=True)
class SearchIndex:
    """A loaded (or failed) CLIP search index.

    ``status is IndexStatus.OK`` guarantees ``embeddings``, ``kf_df`` and
    ``embedder`` are populated and mutually shape-consistent. For
    ``MISSING`` / ``CORRUPT`` the route renders the no-index UI state
    instead of 500-ing; ``detail`` carries a log-friendly reason.

    .. warning::
       ``frozen=True`` prevents *field reassignment* but does NOT
       prevent mutation of the field contents — ``embeddings``
       (numpy array) and ``kf_df`` (pandas DataFrame) are mutable
       objects, and instances of this dataclass are SHARED across
       requests via the module-level cache (:func:`load_index`).
       Callers must treat both fields as read-only. If you need to
       transform the keyframe frame, copy first
       (``index.kf_df.copy()``) — an in-place ``inplace=True`` op or a
       ``kf_df['rank'] = …`` assignment will silently corrupt the
       cached index for every subsequent request on this film.
    """

    status: IndexStatus
    embeddings: object | None = None
    kf_df: object | None = None
    embedder: object | None = None
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status is IndexStatus.OK


# ── mtime/size-aware index cache ──────────────────────────────────────────────

# key: (slug_or_none, embeddings_path_str, mapping_path_str, embedder_name)
# value: (signature, SearchIndex) where signature is the combined stat tuple of
# both files at load time. A changed signature => reload.
# The slug + embedder_name components isolate each film×embedder cache slot.
_index_cache: dict[tuple[str | None, str, str, str], tuple[tuple, SearchIndex]] = {}
_cache_lock = threading.Lock()

# Sibling cache flushers (e.g. BM25 lru_cache) plug in via
# ``register_cache_clearer``. Iterated under no lock — the registry is
# expected to be append-only at import time.
_extra_cache_clearers: list[Callable[[], None]] = []


def register_cache_clearer(fn: Callable[[], None]) -> None:
    """Register a sibling cache to flush from :func:`clear_index_cache`.

    Intended for module-import-time use by sibling caches in the same
    package (e.g. ``cinemateca.search.bm25``) so that the canonical
    test-isolation hook — "one call to ``clear_index_cache()`` drops
    every search-cache layer" — survives the package split.
    """
    _extra_cache_clearers.append(fn)


def _stat_sig(path: Path) -> tuple[int, int] | None:
    """Return ``(st_mtime_ns, st_size)`` for *path*, or ``None`` if absent.

    ``None`` participates in the cache signature so an index file
    appearing or disappearing also invalidates the cached entry.
    """
    try:
        st = path.stat()
    except (FileNotFoundError, NotADirectoryError):
        return None
    return (st.st_mtime_ns, st.st_size)


def _load_and_validate(emb_path: Path, map_path: Path) -> SearchIndex:
    """Load the index from disk and validate its shape coherence.

    Validation done HERE (not in the AI core ``embeddings.py``):

      * either file missing                       -> MISSING
      * unreadable / malformed mapping or npy     -> CORRUPT
      * embeddings row count != keyframe-map rows  -> CORRUPT

    The successful path is byte-equivalent to the old route's
    ``CLIPEmbedder.load`` + ``CLIPEmbedder()`` construction, so a
    well-formed index behaves exactly as before.
    """
    if not emb_path.exists() or not map_path.exists():
        logger.warning("Search index not found at %s", emb_path.parent)
        return SearchIndex(IndexStatus.MISSING, detail="index files absent")

    from cinemateca.models.clip.openclip import OpenClipEmbedder

    # File loader is backend-agnostic: ``OpenClipEmbedder.load`` reads the
    # raw ``.npy`` + JSON mapping and ``SiglipMultilingualEmbedder.save``
    # writes the same on-disk shape, so a SigLIP-produced index is read
    # transparently here. The constructed text-encoder embedder below
    # comes from the registry — the read shape and the live encoder are
    # decoupled on purpose.
    try:
        embeddings, mapping, kf_df = OpenClipEmbedder.load(emb_path, map_path)
    except Exception as exc:  # malformed .npy / .json, missing keys, etc.
        logger.warning("Search index unreadable (%s): %s", emb_path.parent, exc)
        return SearchIndex(IndexStatus.CORRUPT, detail=f"unreadable: {exc}")

    n_emb = int(getattr(embeddings, "shape", [0])[0])
    n_map = len(kf_df)
    declared = mapping.get("total_vectors")
    if n_emb != n_map:
        logger.warning(
            "Corrupt search index: %d embedding rows vs %d keyframe-map " "rows (%s)",
            n_emb,
            n_map,
            emb_path.parent,
        )
        return SearchIndex(
            IndexStatus.CORRUPT,
            detail=f"row mismatch: {n_emb} embeddings vs {n_map} mapping rows",
        )
    if declared is not None and declared != n_map:
        logger.warning(
            "Corrupt search index: mapping declares total_vectors=%s but "
            "has %d keyframe rows (%s)",
            declared,
            n_map,
            emb_path.parent,
        )
        return SearchIndex(
            IndexStatus.CORRUPT,
            detail=(f"mapping total_vectors={declared} != {n_map} keyframe rows"),
        )

    # Use OpenClipEmbedder as the initial default. load_index swaps in the
    # registry-configured backend (e.g. SigLIP) immediately after this
    # function returns when embedder_name != "clip_openclip", so this is
    # always the correct starting point regardless of active backend.
    embedder = OpenClipEmbedder()
    logger.info("Search index loaded: %d vectors", n_map)
    return SearchIndex(IndexStatus.OK, embeddings=embeddings, kf_df=kf_df, embedder=embedder)


def load_index(
    ctx: Any,
    *,
    mapping_filename: str,
    embeddings_filename: str,
    cfg: Any = None,
) -> SearchIndex:
    """Return the (cached) :class:`SearchIndex` for *ctx*'s embeddings dir.

    ``ctx`` is duck-typed: it must expose ``.slug`` (``str | None``) and
    ``.embeddings_dir`` (``Path``). The current single producer is
    :class:`api.services.film_context.FilmContext`; P2 will replace it
    with ``cinemateca.library.Library`` without changing the duck-typed
    surface.

    The result is cached keyed by the embeddings + mapping file paths
    and their ``(mtime_ns, size)`` signature. If either file is changed,
    added or removed since the cached load, the index is reloaded
    transparently — a regenerated index is picked up WITHOUT a process
    restart.

    Always returns a ``SearchIndex`` (never ``None`` / never raises for
    a missing/corrupt index): the route inspects ``.status`` and
    renders the no-index UI state for ``MISSING`` / ``CORRUPT``.
    """
    emb_path = ctx.embeddings_dir / embeddings_filename
    map_path = ctx.embeddings_dir / mapping_filename
    embedder_name = (
        getattr(getattr(cfg, "models", None), "image_embedder", "clip_openclip")
        if cfg is not None
        else "clip_openclip"
    )
    key = (ctx.slug, str(emb_path), str(map_path), embedder_name)
    sig = (_stat_sig(emb_path), _stat_sig(map_path))

    with _cache_lock:
        cached = _index_cache.get(key)
        if cached is not None and cached[0] == sig:
            return cached[1]
        index = _load_and_validate(emb_path, map_path)
        # Swap embedder when a non-default backend is requested. The default
        # path (clip_openclip) keeps the OpenClipEmbedder() already built by
        # _load_and_validate so tests that monkeypatch openclip.OpenClipEmbedder
        # continue to work without any changes.
        if index.ok and embedder_name not in ("clip_openclip", None):
            try:
                from cinemateca.models.registry import get_image_embedder

                alt_embedder = get_image_embedder(cfg)
                index = SearchIndex(
                    status=index.status,
                    embeddings=index.embeddings,
                    kf_df=index.kf_df,
                    embedder=alt_embedder,
                    detail=index.detail,
                )
            except Exception as exc:
                logger.warning("load_index: embedder swap failed (%s), using default", exc)
        _index_cache[key] = (sig, index)
        return index


def clear_index_cache() -> None:
    """Drop every cached index entry + flush sibling-registered caches.

    Production code never needs this — the stat signature handles
    invalidation. ``tests/conftest.py`` calls it per test so a
    populated/corrupt index from one test cannot leak into the next.

    Sibling caches (e.g. the BM25 ``@lru_cache`` in
    :mod:`cinemateca.search.bm25`) register their own flushers via
    :func:`register_cache_clearer`; iterating ``_extra_cache_clearers``
    here keeps the one-call-drops-everything contract intact across the
    package split.
    """
    with _cache_lock:
        _index_cache.clear()
    for fn in _extra_cache_clearers:
        fn()
