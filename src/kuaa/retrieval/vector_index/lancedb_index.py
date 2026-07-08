"""
kuaa.retrieval.vector_index.lancedb_index
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Opt-in VectorIndex backed by an embedded LanceDB table.

LanceDB is serverless and on-disk (no daemon, no node/npm — consistent with
the local-first guardrails). One table with a ``film_slug`` column lets the
cross-film aggregate search collapse from a per-film Python loop into a
single filtered ANN query, and serves the rhymes kNN directly.

Requires the optional ``scale`` extra (``pip install -e ".[scale]"``).
``lancedb`` is imported lazily so the minimal install stays lean and the
default ``numpy_bruteforce`` path never touches it.

NOTE: This backend is wired and unit-testable behind an ``importorskip``
guard, but has not been exercised against a real LanceDB install in this
change — treat it as reviewed-not-run until validated in an environment
with the ``scale`` extra present.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from kuaa.retrieval.vector_index.base import SCORE_COLUMN

logger = logging.getLogger(__name__)

_VECTOR_COLUMN = "vector"
_DISTANCE_COLUMN = "_distance"
_FILM_SLUG_COLUMN = "film_slug"
# Vectors are L2-normalised, so cosine distance = 1 - cosine similarity; we
# invert it back to a similarity score to match the numpy backend's semantics.
_METRIC = "cosine"


def _quote(value: str) -> str:
    """Escape a value for interpolation into a LanceDB SQL ``where`` clause."""
    return value.replace("'", "''")


def _require_lancedb():
    try:
        import lancedb
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "The 'lancedb' backend requires the optional 'scale' extra. "
            'Install it with: pip install -e ".[scale]"'
        ) from exc
    return lancedb


class LanceDBIndex:
    """VectorIndex over a single embedded LanceDB table.

    Args:
        uri: Directory the LanceDB database lives in (created on demand).
        table_name: Table holding all keyframe vectors across films.
    """

    def __init__(self, uri: str | Path, table_name: str = "keyframes") -> None:
        self.uri = str(uri)
        self.table_name = table_name
        self._db = None

    def _connect(self):
        if self._db is None:
            lancedb = _require_lancedb()
            Path(self.uri).mkdir(parents=True, exist_ok=True)
            self._db = lancedb.connect(self.uri)
        return self._db

    def _open_table(self):
        db = self._connect()
        if self.table_name in db.table_names():
            return db.open_table(self.table_name)
        return None

    def _records(self, embeddings: np.ndarray, rows: pd.DataFrame) -> list[dict]:
        emb = np.asarray(embeddings, dtype="float32")
        if len(rows) != emb.shape[0]:
            raise ValueError(f"embeddings/rows row mismatch: {emb.shape[0]} vs {len(rows)}")
        records = rows.to_dict(orient="records")
        for rec, vec in zip(records, emb):
            rec[_VECTOR_COLUMN] = vec.tolist()
        return records

    def add(self, embeddings: np.ndarray, rows: pd.DataFrame) -> None:
        db = self._connect()
        records = self._records(embeddings, rows)
        if not records:
            return
        if self.table_name in db.table_names():
            db.open_table(self.table_name).add(records)
        else:
            db.create_table(self.table_name, data=records)
        logger.info(
            "lancedb: added %d vectors to table %r at %s",
            len(records),
            self.table_name,
            self.uri,
        )

    def delete(self, film_slug: str) -> None:
        table = self._open_table()
        if table is None:
            return
        table.delete(f"{_FILM_SLUG_COLUMN} = '{_quote(film_slug)}'")
        logger.info("lancedb: deleted rows for film_slug=%r from %r", film_slug, self.table_name)

    def search(
        self,
        vector: np.ndarray,
        k: int | None,
        *,
        film_slug: str | None = None,
    ) -> pd.DataFrame:
        table = self._open_table()
        if table is None:
            return pd.DataFrame()

        limit = int(k) if k is not None else int(table.count_rows())
        if limit <= 0:
            return pd.DataFrame()

        query = np.asarray(vector, dtype="float32")
        builder = table.search(query).metric(_METRIC).limit(limit)
        if film_slug is not None:
            # prefilter so the limit applies after the film restriction.
            builder = builder.where(f"{_FILM_SLUG_COLUMN} = '{_quote(film_slug)}'", prefilter=True)

        df = builder.to_pandas()
        if df.empty:
            return df
        # Convert cosine distance back to a similarity score and drop the
        # bulky vector / internal distance columns.
        df[SCORE_COLUMN] = 1.0 - df[_DISTANCE_COLUMN].astype(float)
        df = df.drop(columns=[c for c in (_VECTOR_COLUMN, _DISTANCE_COLUMN) if c in df.columns])
        return df.reset_index(drop=True)

    def knn(self, vector: np.ndarray, k: int) -> pd.DataFrame:
        return self.search(vector, k)
