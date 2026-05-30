"""Unified mtime+size-keyed cache for CLIP / audio / BM25 indexes (C4).

Collapses three duplicated cache implementations (search.cache,
search.audio, search.bm25) onto one ``StatCache``. The cache key's
first element is conventionally the film slug so ``clear_film(slug)``
can invalidate exactly one film's slots across every layer.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

K = TypeVar("K")
V = TypeVar("V")
# Signature is a tuple used only for equality checks — allowing nested
# sub-tuples and None lets the CLIP cache (which carries per-file stat
# tuples that may be None for absent files) use the same type.
Signature = tuple[Any, ...]


def stat_sig(path: Path) -> tuple[int, int] | None:
    """``(st_mtime_ns, st_size)`` for *path*, or ``None`` if absent."""
    try:
        st = path.stat()
    except (FileNotFoundError, NotADirectoryError):
        return None
    return (st.st_mtime_ns, st.st_size)


@runtime_checkable
class CacheKey(Protocol):
    """A cache key whose first component is the film slug (for clear_film)."""

    def __hash__(self) -> int: ...


class StatCache(Generic[K, V]):
    """Thread-safe signature-validated cache with hit/miss counters."""

    def __init__(self) -> None:
        self._store: dict[K, tuple[Signature, V]] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get_or_load(self, *, key: K, signature: Signature, loader: Callable[[], V]) -> V:
        """Return cached value if signature matches; otherwise call ``loader`` and cache the result."""
        with self._lock:
            cached = self._store.get(key)
            if cached is not None and cached[0] == signature:
                self.hits += 1
                return cached[1]
            self.misses += 1
            value = loader()
            self._store[key] = (signature, value)
            return value

    def clear(self) -> None:
        """Evict all cached entries."""
        with self._lock:
            self._store.clear()

    def clear_film(self, slug: str) -> None:
        """Drop every slot whose key's first component equals *slug*."""
        with self._lock:
            doomed = [
                k for k in self._store
                if (k[0] if isinstance(k, tuple) else k) == slug
            ]
            for k in doomed:
                del self._store[k]
