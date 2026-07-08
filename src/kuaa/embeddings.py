"""
kuaa.embeddings
~~~~~~~~~~~~~~~~~~~~~
Busca semântica sobre embeddings CLIP. Delegates the nearest-neighbour
lookup to a :class:`~kuaa.retrieval.vector_index.base.VectorIndex` — the
default ``numpy_bruteforce`` backend is a pure-numpy dot product
(equivalente a cosseno para vetores normalizados), byte-identical to the
former inline ``embeddings @ query`` path. The CLIP embedder itself lives
in kuaa.models.clip.openclip.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from kuaa.retrieval.vector_index.base import SCORE_COLUMN
from kuaa.retrieval.vector_index.numpy_bruteforce import NumpyBruteForceIndex

logger = logging.getLogger(__name__)


class SemanticSearch:
    """Semantic search over CLIP embeddings (text / image / combined).

    The raw vector math runs through an in-memory
    :class:`NumpyBruteForceIndex` (the Protocol's default backend), so the
    ranking is identical to the legacy inline dot-product while the
    similarity computation now lives behind the ``VectorIndex`` seam.
    """

    def __init__(self, embeddings: np.ndarray, keyframes_df: pd.DataFrame, embedder):
        self.embeddings = embeddings
        self.keyframes_df = keyframes_df
        self.embedder = embedder
        # Index over the (non-renormalised) embeddings for by_text / by_image.
        # ``combined`` builds its own transient index over a renormalised
        # subset, matching the historical per-call renormalisation.
        self._index = NumpyBruteForceIndex()
        self._index.add(embeddings, keyframes_df)

    def by_text(self, query: str, top_k: int = 8) -> pd.DataFrame:
        query_emb = self.embedder.encode_text(query)
        hits = self._index.search(query_emb, top_k)

        rows = []
        for rank, (_, row) in enumerate(hits.iterrows()):
            rows.append(
                {
                    "rank": rank + 1,
                    "scene_id": row["scene_id"],
                    "filepath": row["filepath"],
                    "similarity": float(row[SCORE_COLUMN]),
                    "description": str(row.get("description", "")),
                }
            )
        return pd.DataFrame(rows)

    def by_image(
        self,
        image_path: str | Path,
        top_k: int = 8,
        exclude_self: bool = True,
    ) -> pd.DataFrame:
        img_emb = self.embedder.encode_image_single(image_path)

        # Full descending order so exclude-self can drop the query row before
        # the top-k slice (matches the legacy argsort-then-filter sequence).
        hits = self._index.search(img_emb, None)

        if exclude_self and not hits.empty:
            query_str = str(image_path)
            hits = hits[hits["filepath"].astype(str) != query_str]

        hits = hits.head(top_k)

        rows = []
        for rank, (_, row) in enumerate(hits.iterrows()):
            rows.append(
                {
                    "rank": rank + 1,
                    "scene_id": row["scene_id"],
                    "filepath": row["filepath"],
                    "similarity": float(row[SCORE_COLUMN]),
                    "description": str(row.get("description", "")),
                }
            )
        return pd.DataFrame(rows)

    def combined(
        self,
        query: str,
        filter_tags: list[str] | None = None,
        tag_index: dict | None = None,
        top_k: int = 8,
    ) -> pd.DataFrame:
        """
        Busca combinada: filtro por tags LLM + ranking semântico CLIP.

        Args:
            query:        Texto de busca semântica.
            filter_tags:  Lista de tags para pré-filtrar (ex: ["exterior", "dia"]).
            tag_index:    Índice invertido {tag: [scene_ids]} do módulo LLM.
            top_k:        Número de resultados finais.

        Returns:
            DataFrame com colunas: rank, scene_id, similarity, filepath.
        """
        if filter_tags and tag_index:
            # SOLE / REQUIRED normalization for the search path — do not
            # delete believing the caller already normalized. The real
            # caller (api/routes/search.py, around line 123) passes the RAW
            # hybrid index straight in: merge_tag_index(...) untouched. That
            # hybrid mixes int (LLM) and str (manual) scene ids. If this
            # normalize_tag_index call is removed the membership test below
            # silently mismatches and tag-filtered search returns nothing.
            # We normalize the index AND map the df scene_id column to the
            # canonical string key so the test is provably str-vs-str. The
            # .map() is a local computation only — the stored keyframes_df
            # dtype is left untouched (callers downstream read
            # row["scene_id"] for display, which str()-renders identically).
            from kuaa.scene_ids import normalize_tag_index, scene_id_key

            norm_index = normalize_tag_index(tag_index)
            valid_ids = set(norm_index.get(filter_tags[0], set()))
            for tag in filter_tags[1:]:
                valid_ids &= set(norm_index.get(tag, set()))

            # Intentionally per-row via .map(scene_id_key) — NOT a vectorized
            # .astype(str). A NaN-tainted int column is float64, so
            # .astype(str) would yield "351.0" and never match "351",
            # reintroducing the exact bug this code fixes.
            scene_id_keys = self.keyframes_df["scene_id"].map(scene_id_key)
            mask = scene_id_keys.isin(valid_ids)
            kf_subset = self.keyframes_df[mask].reset_index(drop=True)
            emb_subset = self.embeddings[self.keyframes_df[mask].index]
            logger.info("Busca combinada: filtro %s → %d cenas", filter_tags, len(kf_subset))
        else:
            kf_subset = self.keyframes_df.reset_index(drop=True)
            emb_subset = self.embeddings

        if len(kf_subset) == 0:
            logger.warning("Nenhuma cena encontrada com os filtros: %s", filter_tags)
            return pd.DataFrame()

        query_emb = self.embedder.encode_text(query)
        # Re-normalizar subconjunto (por precaução), depois buscar via índice
        # transitório sobre o subconjunto renormalizado (mesma matemática).
        norms = np.linalg.norm(emb_subset, axis=1, keepdims=True) + 1e-8
        emb_norm = emb_subset / norms
        subset_index = NumpyBruteForceIndex()
        subset_index.add(emb_norm, kf_subset)
        hits = subset_index.search(query_emb, top_k)

        rows = []
        for rank, (_, row) in enumerate(hits.iterrows()):
            rows.append(
                {
                    "rank": rank + 1,
                    "scene_id": row["scene_id"],
                    "filepath": row["filepath"],
                    "similarity": float(row[SCORE_COLUMN]),
                }
            )
        return pd.DataFrame(rows)
