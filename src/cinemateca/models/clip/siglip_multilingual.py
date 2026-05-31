"""SigLIP-multilingual visual embedder backend.

Drop-in alternative to :class:`OpenClipEmbedder` selectable via
``cfg.models.image_embedder = "siglip_multilingual"``. The on-disk
artefact shape (``keyframe_embeddings.npy`` + ``index_mapping.json``)
matches the OpenClip backend exactly so the search service and every
downstream reader keep working without changes.

Public surface mirrors :class:`cinemateca.models.clip.openclip.OpenClipEmbedder`:

* ``encode_text(text) -> np.ndarray`` shape ``(D,)`` float32 L2-normalised
* ``encode_image_single(path) -> np.ndarray`` shape ``(D,)``
* ``encode_images(paths) -> np.ndarray`` shape ``(N, D)``
* ``save(embeddings, keyframes_df, output_dir, …) -> (emb_path, map_path)``

The default model id is ``google/siglip2-large-patch16-256`` — a
natively multilingual SigLIP2 checkpoint (current-generation, Feb 2025)
with a 1024-dim projection that overlaps Brazilian Portuguese tokens
out of the box. The original SigLIP only ships multilingual at the
base size; the "large + multilingual" combination lives in the SigLIP2
namespace. Override via ``cfg.embeddings.model_id`` for experimentation.

Task 4.1 of the M3 pre-flight plan ships only the backend + registry
wiring; the default stays ``clip_openclip`` until Task 4.2 (smoke
validation) and Task 4.3 (library re-embed) flip it.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cinemateca.config import Settings
from cinemateca.models.manifest import ModelCard, get_card

logger = logging.getLogger(__name__)

# Imported lazily inside ``_load_model`` so the import-time graph stays
# small and the test suite can monkey-patch these names without paying
# the cost of pulling in ``transformers`` for unrelated tests.
AutoModel: Any = None
AutoProcessor: Any = None

# Serialise the first concurrent ``from transformers import AutoModel`` so
# that two worker threads can't race transformers' ``_LazyModule`` during
# its initial population. uvicorn dispatches search requests through
# ``run_in_executor`` and HTMX commonly fires two near-simultaneous GETs
# for the same query (submit + keyup triggers); without this lock one
# thread sees ``transformers`` partially initialised and raises
# ``ImportError: cannot import name 'AutoModel'`` — fatal for the first
# request but harmless on retry.
_IMPORT_LOCK = threading.Lock()

_DEFAULT_MODEL_ID = "google/siglip2-large-patch16-256"


class SiglipMultilingualEmbedder:
    """SigLIP-multilingual via HuggingFace transformers."""

    #: Provenance for this backend (manifest single source of truth, C10/F6).
    CARD: ModelCard = get_card("siglip_multilingual")

    def __init__(self, cfg: Settings | None = None, device: str | None = None) -> None:
        self._cfg = cfg
        self._device = device
        self._model: Any = None
        self._processor: Any = None

        # Allow the model id to be overridden via cfg.embeddings.model_id;
        # fall back to the multilingual default. The remaining knobs
        # (``batch_size``) live on cfg.embeddings to keep parity with the
        # OpenClip backend, but SigLIP has no separate "pretrained" axis.
        emb = getattr(cfg, "embeddings", None) if cfg is not None else None
        self.model_id = str(getattr(emb, "model_id", _DEFAULT_MODEL_ID))
        self.batch_size = int(getattr(emb, "batch_size", 16)) if emb else 16

    # ------------------------------------------------------------------ load
    def _load_model(self) -> None:
        if self._model is not None:
            return
        global AutoModel, AutoProcessor
        # The whole load path is serialised because (a) concurrent
        # ``from transformers import AutoModel`` races transformers'
        # ``_LazyModule`` and (b) two threads racing the GPU model init
        # would double-allocate weights. Fast path above keeps the steady-
        # state cost a single ``is not None`` check.
        with _IMPORT_LOCK:
            if self._model is not None:
                return
            if AutoModel is None or AutoProcessor is None:
                try:
                    from transformers import AutoModel as _AM
                    from transformers import AutoProcessor as _AP
                except ImportError as exc:  # pragma: no cover - dep guard
                    raise RuntimeError(
                        "SigLIP backend requires 'transformers'. "
                        "Install via: uv sync --extra full"
                    ) from exc
                AutoModel = _AM
                AutoProcessor = _AP

            if self._device is None:
                from cinemateca.device import get_device

                self._device = str(get_device("auto"))

            logger.info(
                "Carregando SigLIP-multilingual %s (device=%s)…",
                self.model_id,
                self._device,
            )
            t0 = time.time()
            # Pin use_fast=False so query-time image processing matches the slow
            # processor used during library indexing. transformers >=4.52 will
            # default to the fast Rust processor, which produces minor pixel-
            # level differences and would desync queries from stored embeddings.
            self._processor = AutoProcessor.from_pretrained(self.model_id, use_fast=False)
            self._model = AutoModel.from_pretrained(self.model_id, device_map=self._device).eval()
            logger.info("✓ SigLIP carregado em %.1fs | device=%s", time.time() - t0, self._device)

    # --------------------------------------------------------------- encoders
    def encode_text(self, text: str) -> np.ndarray:
        """Encode a single text query into a (D,) float32 L2-normalised vector."""
        import torch

        self._load_model()
        with torch.no_grad():
            # SigLIP / SigLIP2 text encoders are trained with a fixed 64-token
            # sequence length; padding="longest" (padding=True) on a short
            # single query produces a 2–3 token input that the model has never
            # seen, yielding noise-level text features (top cosine ~0.05).
            # See https://huggingface.co/docs/transformers/model_doc/siglip2
            inputs = self._processor(
                text=[text],
                return_tensors="pt",
                padding="max_length",
                max_length=64,
                truncation=True,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            feats = self._model.get_text_features(**inputs)
        v = feats.detach().cpu().numpy().astype("float32").squeeze(0)
        n = float(np.linalg.norm(v)) or 1.0
        return (v / n).astype("float32")

    def encode_image_single(self, image_path: str | Path) -> np.ndarray:
        """Encode a single image into a (D,) float32 L2-normalised vector."""
        import torch
        from PIL import Image

        self._load_model()
        img = Image.open(str(image_path)).convert("RGB")
        with torch.no_grad():
            inputs = self._processor(images=[img], return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            feats = self._model.get_image_features(**inputs)
        v = feats.detach().cpu().numpy().astype("float32").squeeze(0)
        n = float(np.linalg.norm(v)) or 1.0
        return (v / n).astype("float32")

    def encode_images(self, image_paths: list[Path]) -> np.ndarray:
        """Encode a list of images into an (N, D) float32 L2-normalised array."""
        import torch
        from PIL import Image

        self._load_model()
        if not image_paths:
            dim = int(getattr(self._model.config, "projection_dim", 1024))
            return np.zeros((0, dim), dtype="float32")

        all_chunks: list[np.ndarray] = []
        error_count = 0
        t0 = time.time()

        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i : i + self.batch_size]
            imgs: list[Image.Image] = []
            for path in batch_paths:
                try:
                    imgs.append(Image.open(str(path)).convert("RGB"))
                except Exception as e:  # noqa: BLE001
                    logger.warning("Erro ao carregar %s: %s", path, e)
                    # Fall back to a black placeholder so the batch shape
                    # is preserved; the row will be a near-zero vector.
                    imgs.append(Image.new("RGB", (256, 256), color=(0, 0, 0)))
                    error_count += 1

            with torch.no_grad():
                inputs = self._processor(images=imgs, return_tensors="pt")
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                feats = self._model.get_image_features(**inputs)
            all_chunks.append(feats.detach().cpu().numpy().astype("float32"))

            if (i // self.batch_size + 1) % 10 == 0:
                logger.info(
                    "SigLIP embeddings: %d/%d imagens processadas",
                    min(i + self.batch_size, len(image_paths)),
                    len(image_paths),
                )

        embeddings = np.vstack(all_chunks).astype("float32")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        embeddings = (embeddings / norms).astype("float32")
        logger.info(
            "✓ %d embeddings SigLIP gerados em %.1fs (erros: %d) | shape=%s",
            len(image_paths),
            time.time() - t0,
            error_count,
            embeddings.shape,
        )
        return embeddings

    # ----------------------------------------------------------------- save/load
    def save(
        self,
        embeddings: np.ndarray,
        keyframes_df: pd.DataFrame,
        output_dir: str | Path,
        embeddings_filename: str = "keyframe_embeddings.npy",
        mapping_filename: str = "index_mapping.json",
    ) -> tuple[Path, Path]:
        """Persist embeddings + mapping. Mirrors OpenClipEmbedder.save shape."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        emb_path = out / embeddings_filename
        np.save(emb_path, embeddings.astype("float32"))
        logger.info(
            "✓ Embeddings salvos: %s | %.1f MB",
            emb_path,
            emb_path.stat().st_size / 1e6,
        )

        mapping = {
            "model": f"SigLIP-multilingual ({self.model_id})",
            "dimension": int(embeddings.shape[1]) if embeddings.size else 0,
            "total_vectors": int(len(embeddings)),
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
        """Symmetric loader — same shape as OpenClipEmbedder.load."""
        emb = np.load(embeddings_path)
        with open(mapping_path, encoding="utf-8") as f:
            mapping = json.load(f)

        kf_df = pd.DataFrame(
            {
                "filepath": mapping["keyframe_paths"],
                "scene_id": mapping["scene_ids"],
            }
        )
        if "keyframe_ids" in mapping:
            kf_df["keyframe_id"] = mapping["keyframe_ids"]

        logger.info(
            "✓ Embeddings carregados: shape=%s | %d keyframes mapeados",
            emb.shape,
            len(kf_df),
        )
        return emb, mapping, kf_df
