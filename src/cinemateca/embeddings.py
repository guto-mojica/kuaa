"""
cinemateca.embeddings
~~~~~~~~~~~~~~~~~~~~~
Busca semântica sobre embeddings CLIP. Pure-numpy dot product
(equivalente a cosseno para vetores normalizados). The CLIP embedder
itself lives in cinemateca.models.clip.openclip.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SemanticSearch:
    """Semantic search over CLIP embeddings (text / image / combined)."""

    def __init__(self, embeddings: np.ndarray, keyframes_df: pd.DataFrame, embedder):
        self.embeddings = embeddings
        self.keyframes_df = keyframes_df
        self.embedder = embedder

    def by_text(self, query: str, top_k: int = 8) -> pd.DataFrame:
        query_emb = self.embedder.encode_text(query)
        similarities = (self.embeddings @ query_emb).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]

        rows = []
        for rank, idx in enumerate(top_indices):
            row = self.keyframes_df.iloc[idx]
            rows.append({
                "rank": rank + 1,
                "scene_id": row["scene_id"],
                "filepath": row["filepath"],
                "similarity": float(similarities[idx]),
            })
        return pd.DataFrame(rows)

    def by_image(
        self,
        image_path,
        top_k: int = 8,
        exclude_self: bool = True,
    ) -> pd.DataFrame:
        img_emb = self.embedder.encode_image_single(image_path)

        similarities = (self.embeddings @ img_emb).flatten()
        sorted_indices: list[int] = list(np.argsort(similarities)[::-1])

        if exclude_self:
            query_str = str(image_path)
            sorted_indices = [
                i for i in sorted_indices
                if str(self.keyframes_df.iloc[i]["filepath"]) != query_str
            ]

        top_indices = sorted_indices[:top_k]

        rows = []
        for rank, idx in enumerate(top_indices):
            row = self.keyframes_df.iloc[idx]
            rows.append({
                "rank": rank + 1,
                "scene_id": row["scene_id"],
                "filepath": row["filepath"],
                "similarity": float(similarities[idx]),
            })
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
            # delete believing the caller already normalized. Every real
            # caller passes the RAW hybrid index straight in:
            # api/routes/search.py (~L123) hands over merge_tag_index(...)
            # untouched, and app_streamlit.py (~L255) does the same. That
            # hybrid mixes int (LLM) and str (manual) scene ids. If this
            # normalize_tag_index call is removed the membership test below
            # silently mismatches and tag-filtered search returns nothing.
            # We normalize the index AND map the df scene_id column to the
            # canonical string key so the test is provably str-vs-str. The
            # .map() is a local computation only — the stored keyframes_df
            # dtype is left untouched (callers downstream read
            # row["scene_id"] for display, which str()-renders identically).
            from cinemateca.scene_ids import normalize_tag_index, scene_id_key

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
            logger.info(
                "Busca combinada: filtro %s → %d cenas", filter_tags, len(kf_subset)
            )
        else:
            kf_subset = self.keyframes_df.reset_index(drop=True)
            emb_subset = self.embeddings

        if len(kf_subset) == 0:
            logger.warning("Nenhuma cena encontrada com os filtros: %s", filter_tags)
            return pd.DataFrame()

        query_emb = self.embedder.encode_text(query)
        # Re-normalizar subconjunto (por precaução)
        norms = np.linalg.norm(emb_subset, axis=1, keepdims=True) + 1e-8
        emb_norm = emb_subset / norms
        similarities = (emb_norm @ query_emb).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]

        rows = []
        for rank, idx in enumerate(top_indices):
            row = kf_subset.iloc[idx]
            rows.append({
                "rank": rank + 1,
                "scene_id": row["scene_id"],
                "filepath": row["filepath"],
                "similarity": float(similarities[idx]),
            })
        return pd.DataFrame(rows)
