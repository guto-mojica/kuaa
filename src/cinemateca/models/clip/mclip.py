"""M-CLIP multilingual text-encoder backend.

Overrides only ``encode_text()`` — image encoding is inherited unchanged
from :class:`OpenClipEmbedder` (ViT-B/32 openai weights). This is the key
property of M-CLIP: the text encoder is retrained to produce vectors in the
same 512-dim space as the original CLIP ViT-B/32 image encoder, so existing
``keyframe_embeddings.npy`` need no regeneration.

Text model: ``clip-ViT-B-32-multilingual-v1`` (sentence-transformers hub).
This is a knowledge-distilled multilingual variant of CLIP ViT-B/32,
supporting 50+ languages including Portuguese. It outputs L2-normalised
512-dim vectors directly comparable to the ViT-B/32 image embeddings already
stored in the library.

No additional package dependency: ``sentence-transformers`` is already in the
``full`` extra (added for the cross-encoder reranker in M2).
"""

from __future__ import annotations

import logging
import time

import numpy as np

from cinemateca.models.clip.openclip import OpenClipEmbedder

logger = logging.getLogger(__name__)

_ST_MODEL_NAME = "clip-ViT-B-32-multilingual-v1"


class MClipEmbedder(OpenClipEmbedder):
    """M-CLIP multilingual image+text embedder.

    ``encode_text()`` uses the multilingual XLM-Roberta text encoder aligned
    with CLIP ViT-B/32 visual space. All image methods (``encode_images``,
    ``encode_image_single``) are inherited from :class:`OpenClipEmbedder`
    without modification.
    """

    def __init__(self, cfg=None, device=None):
        super().__init__(cfg, device)
        self._st_model = None

    def _load_mclip(self) -> None:
        if self._st_model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise RuntimeError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )
        logger.info("Loading M-CLIP text encoder %r…", _ST_MODEL_NAME)
        t0 = time.time()
        # Force CPU: sentence-transformers auto-detects MPS on Apple Silicon,
        # but MPS is not thread-safe. The search path runs in a thread executor,
        # and a single text encode is fast enough on CPU.
        self._st_model = SentenceTransformer(_ST_MODEL_NAME, device="cpu")
        logger.info("✓ M-CLIP loaded in %.1fs | device=cpu", time.time() - t0)

    def encode_text(self, text: str) -> np.ndarray:
        """Return (512,) float32 L2-normalised vector via multilingual encoder."""
        self._load_mclip()
        vec = self._st_model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vec[0].astype("float32")
