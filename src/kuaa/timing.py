"""Lightweight timing hook (F5).

A context manager that measures wall-clock elapsed time in milliseconds
and logs it at DEBUG. Consumed by the search dispatchers (WS-1 C9, which
attaches ``latency_ms`` to ``SearchResult``) and the benchmark harness
(WS-4 E6 p50/p95/p99). Zero dependencies beyond stdlib.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Timer:
    """Holds the measured elapsed time. ``elapsed_ms`` is frozen at exit."""

    label: str | None = None
    elapsed_ms: float = 0.0


@contextmanager
def timed(label: str | None = None) -> Iterator[Timer]:
    """Measure the wrapped block's wall-clock duration in milliseconds.

    Usage::

        with timed("search.encode") as t:
            ...
        result.latency_ms = t.elapsed_ms
    """
    t = Timer(label=label)
    start = time.perf_counter()
    try:
        yield t
    finally:
        t.elapsed_ms = (time.perf_counter() - start) * 1000.0
        if label:
            logger.debug("timed %s: %.2f ms", label, t.elapsed_ms)


__all__ = ["Timer", "timed"]
