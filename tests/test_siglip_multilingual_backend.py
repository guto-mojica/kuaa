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
    from kuaa.models.clip.siglip_multilingual import (
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

    from kuaa.models.clip import siglip_multilingual as mod

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


def test_siglip_encode_text_uses_fixed_length_padding(monkeypatch):
    """Regression: encode_text must pad to SigLIP's fixed 64-token length.

    SigLIP / SigLIP2 text encoders are trained with sequences padded to
    64 tokens; padding="longest" (padding=True) on a short query produces
    a 2–3 token input that the model never saw, yielding noise-level text
    features (top cosine ~0.05 instead of ~0.12). Lock the kwargs in.
    """
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    import torch

    from kuaa.models.clip import siglip_multilingual as mod

    captured: dict = {}

    class _StubModel:
        config = type("Cfg", (), {"projection_dim": 768})

        def to(self, *_a, **_k):
            return self

        def eval(self):
            return self

        def get_text_features(self, **_inputs):
            return torch.ones((1, 768))

    class _StubProc:
        def __call__(self, *args, **kwargs):
            captured.update(kwargs)
            return {"input_ids": torch.tensor([[1] * 64])}

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
    embedder.encode_text("dog")

    assert captured.get("padding") == "max_length", (
        f"expected padding='max_length' (SigLIP2 fixed 64-token contract), "
        f"got {captured.get('padding')!r}"
    )
    assert captured.get("max_length") == 64
    assert captured.get("truncation") is True


def test_siglip_load_model_serialises_concurrent_calls(monkeypatch):
    """Regression: two worker threads calling _load_model() concurrently must
    not race transformers' ``_LazyModule`` nor double-allocate the GPU model.

    HTMX in this app fires near-simultaneous GETs from ``submit`` + ``keyup``
    triggers; uvicorn dispatches each through ``run_in_executor``. Without
    serialisation, one thread sees a partially initialised transformers
    package and raises ``ImportError: cannot import name 'AutoModel'``.
    """
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    import threading
    import time as _time

    from kuaa.models.clip import siglip_multilingual as mod

    call_count = {"model": 0, "proc": 0}

    class _StubModel:
        config = type("Cfg", (), {"projection_dim": 8})

        def to(self, *_a, **_k):
            return self

        def eval(self):
            return self

    class _StubProc:
        pass

    def _slow_model(*_a, **_k):
        _time.sleep(0.05)  # widen the race window so an unlocked impl loses
        call_count["model"] += 1
        return _StubModel()

    def _slow_proc(*_a, **_k):
        _time.sleep(0.05)
        call_count["proc"] += 1
        return _StubProc()

    monkeypatch.setattr(
        mod,
        "AutoModel",
        type("M", (), {"from_pretrained": staticmethod(_slow_model)}),
    )
    monkeypatch.setattr(
        mod,
        "AutoProcessor",
        type("P", (), {"from_pretrained": staticmethod(_slow_proc)}),
    )

    embedder = mod.SiglipMultilingualEmbedder(cfg=None, device="cpu")
    errors: list[BaseException] = []

    def _worker() -> None:
        try:
            embedder._load_model()
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=_worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"concurrent _load_model raised: {errors!r}"
    assert call_count["model"] == 1, (
        f"AutoModel.from_pretrained called {call_count['model']}× — race not serialised"
    )
    assert call_count["proc"] == 1, (
        f"AutoProcessor.from_pretrained called {call_count['proc']}× — race not serialised"
    )
