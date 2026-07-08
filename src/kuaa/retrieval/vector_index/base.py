"""
kuaa.retrieval.vector_index.base
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Protocol for nearest-neighbour search over keyframe embeddings.

The pipeline imports only from here / the registry — never from a concrete
backend — so swapping ``numpy_bruteforce`` ↔ ``lancedb`` is a config change.

``SCORE_COLUMN`` is the cosine-similarity column every backend appends to
the rows it returns; consumers read it (not a raw numpy array) so the same
downstream code works whether the neighbours came from an in-memory matmul
or an on-disk ANN query.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

#: Name of the cosine-similarity column appended to search/knn results.
SCORE_COLUMN = "score"


@runtime_checkable
class VectorIndex(Protocol):
    """A searchable store of L2-normalised keyframe embeddings + their rows.

    Rows are the per-keyframe metadata (``scene_id`` / ``filepath`` / …, and
    optionally ``film_slug`` for a cross-film index). ``add`` ingests an
    ``(N, D)`` embedding matrix aligned row-for-row with an ``N``-row
    DataFrame; ``search`` / ``knn`` return a DataFrame of the matching rows
    with a :data:`SCORE_COLUMN` column, ordered by descending cosine
    similarity.
    """

    def add(self, embeddings: np.ndarray, rows: pd.DataFrame) -> None:
        """Ingest ``(N, D)`` embeddings aligned to an ``N``-row DataFrame."""
        ...

    def delete(self, film_slug: str) -> None:
        """Remove every row tagged with ``film_slug``, if any.

        No-op when the index has no ``film_slug`` column (or is empty) —
        callers that want an idempotent replace call this immediately
        before :meth:`add` for the same slug (see
        ``kuaa library reindex-vectors``), so re-running a backfill doesn't
        accumulate duplicate vectors.
        """
        ...

    def search(
        self,
        vector: np.ndarray,
        k: int | None,
        *,
        film_slug: str | None = None,
    ) -> pd.DataFrame:
        """Return the top-``k`` rows by cosine similarity to ``vector``.

        ``k=None`` returns every row, ordered by descending similarity (used
        by callers that must post-filter — e.g. exclude-self — before
        trimming). ``film_slug`` restricts the search to one film when the
        index carries a ``film_slug`` column (ignored otherwise).
        """
        ...

    def knn(self, vector: np.ndarray, k: int) -> pd.DataFrame:
        """Return the ``k`` nearest neighbours across the whole index.

        Semantically ``search(vector, k)`` with no film filter — the name
        marks the cross-film visual-rhymes use site.
        """
        ...
