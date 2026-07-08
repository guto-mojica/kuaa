"""
kuaa.retrieval.vector_index.numpy_bruteforce
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Default VectorIndex: in-memory brute-force cosine search.

Reproduces the historical ``(embeddings @ query).flatten()`` +
``np.argsort(sims)[::-1]`` exactly, so results are byte-identical to the
pre-seam :class:`kuaa.embeddings.SemanticSearch` and the per-film rhymes
loop. Zero new dependencies; suitable for archive scale (hours → thousands
of keyframes). The on-disk store for this backend remains the existing
``keyframe_embeddings.npy`` + ``index_mapping.json`` pair.

``add`` deliberately performs NO row-count consistency check, matching the
AI core's historical contract: index-shape validation lives one layer up,
in ``kuaa.search.cache.load_index`` (see that module's docstring). A
mismatched index is accepted silently here and only crashes later, at
``.iloc`` inside :meth:`search`, exactly where the pre-seam inline code
crashed — so callers that rely on that failure mode (and the service-layer
guard that pre-empts it) see unchanged behaviour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kuaa.retrieval.vector_index.base import SCORE_COLUMN

_FILM_SLUG_COLUMN = "film_slug"


class NumpyBruteForceIndex:
    """Brute-force cosine index over an in-memory ``(N, D)`` matrix.

    ``add`` may be called multiple times; each call concatenates onto the
    existing matrix and rows (so a cross-film index accrues one film at a
    time, exactly like the rhymes loop reads one film at a time).
    """

    def __init__(self) -> None:
        self._emb: np.ndarray | None = None
        self._rows: pd.DataFrame = pd.DataFrame()

    def add(self, embeddings: np.ndarray, rows: pd.DataFrame) -> None:
        emb = np.asarray(embeddings)
        if self._emb is None:
            self._emb = emb
            self._rows = rows.reset_index(drop=True)
        else:
            self._emb = np.vstack([self._emb, emb])
            self._rows = pd.concat([self._rows, rows], ignore_index=True)

    def delete(self, film_slug: str) -> None:
        if self._emb is None or _FILM_SLUG_COLUMN not in self._rows.columns:
            return
        keep = (self._rows[_FILM_SLUG_COLUMN] != film_slug).to_numpy()
        self._emb = self._emb[keep]
        self._rows = self._rows[keep].reset_index(drop=True)

    def search(
        self,
        vector: np.ndarray,
        k: int | None,
        *,
        film_slug: str | None = None,
    ) -> pd.DataFrame:
        if self._emb is None or len(self._rows) == 0:
            return pd.DataFrame()

        emb = self._emb
        rows = self._rows
        if film_slug is not None and _FILM_SLUG_COLUMN in rows.columns:
            mask = (rows[_FILM_SLUG_COLUMN] == film_slug).to_numpy()
            emb = emb[mask]
            rows = rows[mask].reset_index(drop=True)
            if len(rows) == 0:
                return pd.DataFrame()

        # Byte-identical to the legacy path: dot product against the
        # L2-normalised matrix (cosine for unit vectors), descending argsort.
        similarities = (emb @ vector).flatten()
        order = np.argsort(similarities)[::-1]
        if k is not None:
            order = order[:k]

        out = rows.iloc[order].reset_index(drop=True)
        out[SCORE_COLUMN] = similarities[order].astype(float)
        return out

    def knn(self, vector: np.ndarray, k: int) -> pd.DataFrame:
        return self.search(vector, k)
