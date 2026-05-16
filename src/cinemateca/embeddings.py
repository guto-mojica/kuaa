"""
cinemateca.embeddings
~~~~~~~~~~~~~~~~~~~~~
Geração de embeddings visuais com CLIP e busca semântica por texto ou imagem.

Baseado no Notebook 04 (04_embeddings_busca.ipynb).
Não requer FAISS — usa produto escalar NumPy (equivalente a cosine similarity
para embeddings normalizados).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)


class CLIPEmbedder:
    """
    Gera embeddings de imagens e textos usando CLIP (ViT-B/32 via open_clip).

    Os embeddings são normalizados (norma L2 = 1) para que o produto escalar
    seja equivalente à similaridade de cosseno — sem necessidade de FAISS.

    Exemplo:
        embedder = CLIPEmbedder(cfg, device)
        embeddings = embedder.encode_images(keyframe_paths)
        embedder.save(embeddings, keyframes_df, cfg.paths.embeddings_dir)
    """

    def __init__(self, cfg=None, device=None):
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._device = device

        if cfg is not None:
            emb = cfg.embeddings
            self.model_name = emb.model
            self.pretrained = emb.pretrained
            self.batch_size = emb.batch_size
        else:
            self.model_name = "ViT-B-32"
            self.pretrained = "openai"
            self.batch_size = 16

    def _load_model(self):
        if self._model is not None:
            return
        try:
            import open_clip
        except ImportError:
            raise RuntimeError(
                "open_clip não instalado. Execute: pip install open-clip-torch"
            )

        import torch

        logger.info("Carregando CLIP %s (%s)...", self.model_name, self.pretrained)
        t0 = time.time()

        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained
        )
        self._tokenizer = open_clip.get_tokenizer(self.model_name)
        self._model = self._model.to(self._device)
        self._model.eval()

        logger.info("✓ CLIP carregado em %.1fs | device=%s", time.time() - t0, self._device)

    def encode_images(self, image_paths: List[Path]) -> np.ndarray:
        """
        Gera embeddings normalizados para uma lista de imagens.

        Processa em batches para eficiência. Imagens que falharem ao carregar
        recebem um vetor zerado como placeholder (mantendo alinhamento com índices).

        Args:
            image_paths: Lista de Paths das imagens.

        Returns:
            np.ndarray de shape (N, 512), dtype float32, L2-normalizado.
        """
        import torch
        import torch.nn.functional as F

        self._load_model()

        all_embeddings = []
        error_count = 0
        t0 = time.time()

        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i: i + self.batch_size]
            tensors = []

            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    tensors.append(self._preprocess(img))
                except Exception as e:
                    logger.warning("Erro ao carregar %s: %s", path, e)
                    tensors.append(torch.zeros(3, 224, 224))
                    error_count += 1

            batch = torch.stack(tensors).to(self._device)
            with torch.no_grad():
                feats = self._model.encode_image(batch)
                feats = F.normalize(feats, dim=-1)
            all_embeddings.append(feats.cpu().numpy())

            if (i // self.batch_size + 1) % 10 == 0:
                logger.info(
                    "Embeddings: %d/%d imagens processadas",
                    min(i + self.batch_size, len(image_paths)),
                    len(image_paths),
                )

        embeddings = np.vstack(all_embeddings).astype("float32")
        elapsed = time.time() - t0
        logger.info(
            "✓ %d embeddings gerados em %.1fs (erros: %d) | shape=%s",
            len(image_paths),
            elapsed,
            error_count,
            embeddings.shape,
        )
        return embeddings

    def encode_text(self, text: str) -> np.ndarray:
        """
        Converte uma string de texto em embedding CLIP normalizado.

        Args:
            text: Texto de busca (inglês recomendado para CLIP padrão).

        Returns:
            np.ndarray de shape (512,), float32, L2-normalizado.
        """
        import torch
        import torch.nn.functional as F

        self._load_model()

        tokens = self._tokenizer([text]).to(self._device)
        with torch.no_grad():
            feat = self._model.encode_text(tokens)
            feat = F.normalize(feat, dim=-1)
        return feat.cpu().numpy().astype("float32")[0]

    def save(
        self,
        embeddings: np.ndarray,
        keyframes_df: pd.DataFrame,
        output_dir: str | Path,
        embeddings_filename: str = "keyframe_embeddings.npy",
        mapping_filename: str = "index_mapping.json",
    ) -> tuple[Path, Path]:
        """
        Persiste embeddings (.npy) e mapeamento índice→metadados (.json).

        Args:
            embeddings:         Matriz gerada por encode_images().
            keyframes_df:       DataFrame com colunas filepath, scene_id, etc.
            output_dir:         Diretório de destino.
            embeddings_filename: Nome do arquivo .npy.
            mapping_filename:   Nome do arquivo .json.

        Returns:
            Tupla (embeddings_path, mapping_path).
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        emb_path = out / embeddings_filename
        np.save(emb_path, embeddings)
        logger.info("✓ Embeddings salvos: %s | %.1f MB", emb_path, emb_path.stat().st_size / 1e6)

        mapping = {
            "model": f"CLIP {self.model_name} ({self.pretrained})",
            "dimension": embeddings.shape[1],
            "total_vectors": len(embeddings),
            "normalized": True,
            "keyframe_paths": keyframes_df["filepath"].tolist(),
            "scene_ids": keyframes_df["scene_id"].tolist(),
        }
        # Incluir keyframe_id se disponível
        if "keyframe_id" in keyframes_df.columns:
            mapping["keyframe_ids"] = keyframes_df["keyframe_id"].tolist()

        map_path = out / mapping_filename
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)
        logger.info("✓ Mapeamento salvo: %s", map_path)

        return emb_path, map_path

    @staticmethod
    def load(
        embeddings_path: str | Path,
        mapping_path: str | Path,
    ) -> tuple[np.ndarray, dict, pd.DataFrame]:
        """
        Carrega embeddings e mapeamento do disco.

        Returns:
            Tupla (embeddings_array, mapping_dict, keyframes_df).
        """
        emb = np.load(embeddings_path)
        with open(mapping_path, encoding="utf-8") as f:
            mapping = json.load(f)

        kf_df = pd.DataFrame({
            "filepath": mapping["keyframe_paths"],
            "scene_id": mapping["scene_ids"],
        })
        if "keyframe_ids" in mapping:
            kf_df["keyframe_id"] = mapping["keyframe_ids"]

        logger.info(
            "✓ Embeddings carregados: shape=%s | %d keyframes mapeados",
            emb.shape,
            len(kf_df),
        )
        return emb, mapping, kf_df


# ─── Motor de Busca ───────────────────────────────────────────────────────────

class SemanticSearch:
    """
    Busca semântica sobre os embeddings CLIP gerados por CLIPEmbedder.

    Suporta busca por texto e busca por imagem de referência.
    Não requer dependências externas além de NumPy.

    Exemplo:
        search = SemanticSearch(embeddings, keyframes_df, embedder)
        results = search.by_text("two people talking outdoors", top_k=8)
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        keyframes_df: pd.DataFrame,
        embedder: CLIPEmbedder,
    ):
        self.embeddings = embeddings
        self.keyframes_df = keyframes_df
        self.embedder = embedder

    def by_text(self, query: str, top_k: int = 8) -> pd.DataFrame:
        """
        Busca keyframes por similaridade semântica com um texto.

        Args:
            query:  Texto de busca (inglês recomendado para CLIP).
            top_k:  Número de resultados a retornar.

        Returns:
            DataFrame com colunas: rank, scene_id, filepath, similarity.
        """
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
        image_path: str | Path,
        top_k: int = 8,
        exclude_self: bool = True,
    ) -> pd.DataFrame:
        """
        Busca keyframes visualmente similares a uma imagem de referência.

        Args:
            image_path:   Caminho da imagem de referência.
            top_k:        Número de resultados.
            exclude_self: Se True, exclui o próprio frame (sim=1.0).

        Returns:
            DataFrame com colunas: rank, scene_id, filepath, similarity.
        """
        import torch
        import torch.nn.functional as F

        self.embedder._load_model()

        img = Image.open(image_path).convert("RGB")
        tensor = self.embedder._preprocess(img).unsqueeze(0).to(self.embedder._device)

        with torch.no_grad():
            feat = self.embedder._model.encode_image(tensor)
            feat = F.normalize(feat, dim=-1)
        img_emb = feat.cpu().numpy().astype("float32")[0]

        similarities = (self.embeddings @ img_emb).flatten()
        top_indices = np.argsort(similarities)[::-1]

        if exclude_self:
            query_str = str(image_path)
            top_indices = [
                i for i in top_indices
                if str(self.keyframes_df.iloc[i]["filepath"]) != query_str
            ]

        top_indices = top_indices[:top_k]

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
        filter_tags: Optional[List[str]] = None,
        tag_index: Optional[dict] = None,
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
