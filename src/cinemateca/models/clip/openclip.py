"""open_clip image/text embedder backend (moved from embeddings.py)."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)


class OpenClipEmbedder:
    """CLIP (ViT-B/32 via open_clip) image/text embedder, L2-normalised."""

    def __init__(self, cfg=None, device=None):
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._device = device

        emb = getattr(cfg, "embeddings", None) if cfg is not None else None
        if emb is not None:
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

        logger.info("Carregando CLIP %s (%s)...", self.model_name, self.pretrained)
        t0 = time.time()

        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained
        )
        self._tokenizer = open_clip.get_tokenizer(self.model_name)
        self._model = self._model.to(self._device)
        self._model.eval()

        logger.info("✓ CLIP carregado em %.1fs | device=%s", time.time() - t0, self._device)

    def encode_images(self, image_paths: list[Path]) -> np.ndarray:
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
        logger.info(
            "✓ %d embeddings gerados em %.1fs (erros: %d) | shape=%s",
            len(image_paths), time.time() - t0, error_count, embeddings.shape,
        )
        return embeddings

    def encode_text(self, text: str) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        self._load_model()

        tokens = self._tokenizer([text]).to(self._device)
        with torch.no_grad():
            feat = self._model.encode_text(tokens)
            feat = F.normalize(feat, dim=-1)
        return feat.cpu().numpy().astype("float32")[0]

    def encode_image_single(self, image_path: str | Path) -> np.ndarray:
        """One image -> (D,) float32 L2-normalised vector (image search).

        Extracted from the old SemanticSearch.by_image inline logic;
        math is identical (same preprocess + encode_image + L2 normalise).
        """
        import torch
        import torch.nn.functional as F

        self._load_model()

        img = Image.open(image_path).convert("RGB")
        tensor = self._preprocess(img).unsqueeze(0).to(self._device)
        with torch.no_grad():
            feat = self._model.encode_image(tensor)
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
            emb.shape, len(kf_df),
        )
        return emb, mapping, kf_df
