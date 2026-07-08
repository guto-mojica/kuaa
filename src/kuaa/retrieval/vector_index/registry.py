"""
kuaa.retrieval.vector_index.registry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Factory that constructs the configured VectorIndex backend.

Reads ``cfg.search.index_backend`` and returns the matching backend,
mirroring ``kuaa.models.registry``: the pipeline imports only from here, so
selecting ``lancedb`` over the default ``numpy_bruteforce`` is a config
change. Concrete backends are imported lazily so the minimal install never
imports the optional ``lancedb`` dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kuaa.config import Settings
    from kuaa.retrieval.vector_index.base import VectorIndex

#: Table directory created under ``paths.library_dir`` for the lancedb backend.
_LANCEDB_DIRNAME = "vector_index.lancedb"


def _backend_name(cfg: Settings) -> str:
    search = getattr(cfg, "search", None)
    return getattr(search, "index_backend", "numpy_bruteforce") if search else "numpy_bruteforce"


def get_vector_index(cfg: Settings, *, uri: str | Path | None = None) -> VectorIndex:
    """Return the configured vector-index backend.

    Args:
        cfg: Loaded settings; ``cfg.search.index_backend`` selects the backend.
        uri: Optional on-disk location for persistent backends (lancedb).
            Defaults to ``cfg.paths.library_dir / vector_index.lancedb``.
    """
    name = _backend_name(cfg)
    if name == "numpy_bruteforce":
        from kuaa.retrieval.vector_index.numpy_bruteforce import NumpyBruteForceIndex

        return NumpyBruteForceIndex()
    if name == "lancedb":
        from kuaa.retrieval.vector_index.lancedb_index import LanceDBIndex

        if uri is None:
            uri = Path(cfg.paths.library_dir) / _LANCEDB_DIRNAME
        return LanceDBIndex(uri)
    raise ValueError(f"Unknown index_backend: {name!r}")
