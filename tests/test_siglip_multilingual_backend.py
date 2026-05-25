"""SigLIP-multilingual backend smoke test (skipped without HF deps).

Task 4.1 of the M3 pre-flight plan. The SigLIP-multilingual backend is a
drop-in alternative to ``OpenClipEmbedder`` exposed via
``cfg.models.image_embedder = "siglip_multilingual"``. The default stays
``clip_openclip`` — flipping is the responsibility of Task 4.2/4.3.

These tests never download the ~2 GB SigLIP weights:

* ``test_siglip_backend_has_expected_protocol_methods`` checks the class
  surface only (no instantiation, no model load).
* ``test_siglip_backend_text_encoding_l2_normalised`` stubs
  ``AutoModel`` / ``AutoProcessor`` on the module so ``encode_text``
  exercises the full L2-normalise path against fake tensors.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_siglip_backend_has_expected_protocol_methods():
    """Class surface mirrors OpenClipEmbedder's public methods."""
    from cinemateca.models.clip.siglip_multilingual import (
        SiglipMultilingualEmbedder,
    )

    cls = SiglipMultilingualEmbedder
    for name in ("encode_text", "encode_image_single", "encode_images", "save"):
        assert hasattr(cls, name), f"Missing required method: {name}"


def test_siglip_backend_text_encoding_l2_normalised(monkeypatch):
    """encode_text returns an L2-normalised (D,) float32 vector.

    Stubs ``AutoModel`` / ``AutoProcessor`` so no weights are pulled.
    Skipped if ``transformers`` isn't installed in the dev env.
    """
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    import torch

    from cinemateca.models.clip import siglip_multilingual as mod

    class _StubModel:
        config = type("Cfg", (), {"projection_dim": 768})

        def to(self, *_args, **_kwargs):
            return self

        def eval(self):
            return self

        def get_text_features(self, **_inputs):
            # Unnormalised vector — backend must L2-normalise.
            vec = [0.0, 0.5, 0.0, 0.5] + [0.0] * 764
            return torch.tensor([vec])

    class _StubProc:
        def __call__(self, *_args, **_kwargs):
            return {"input_ids": torch.tensor([[1, 2, 3]])}

    monkeypatch.setattr(
        mod,
        "AutoModel",
        type("M", (), {"from_pretrained": staticmethod(lambda *_a, **_k: _StubModel())}),
    )
    monkeypatch.setattr(
        mod,
        "AutoProcessor",
        type("P", (), {"from_pretrained": staticmethod(lambda *_a, **_k: _StubProc())}),
    )

    embedder = mod.SiglipMultilingualEmbedder(cfg=None, device="cpu")
    v = embedder.encode_text("um cachorro correndo")

    assert v.shape == (768,)
    assert v.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(v), 1.0, atol=1e-5)
