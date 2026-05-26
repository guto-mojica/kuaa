"""Unit tests for MClipEmbedder — all model loading is monkeypatched."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cinemateca.models.clip.mclip import MClipEmbedder, _ST_MODEL_NAME


# ── helpers ────────────────────────────────────────────────────────────────────

def _fake_st(dim: int = 512) -> MagicMock:
    """Mock SentenceTransformer that encodes texts to fixed-shape arrays."""
    st = MagicMock()
    st.encode.side_effect = lambda texts, **kw: np.ones((len(texts), dim), dtype="float32")
    return st


def _embedder_with_st(fake_st=None, device=None) -> MClipEmbedder:
    """Return an MClipEmbedder with _st_model pre-set, bypassing network load."""
    emb = MClipEmbedder(cfg=None, device=device)
    emb._st_model = fake_st if fake_st is not None else _fake_st()
    return emb


# ── encode_text ────────────────────────────────────────────────────────────────

def test_encode_text_shape_and_dtype():
    emb = _embedder_with_st()
    vec = emb.encode_text("homem no campo")
    assert vec.shape == (512,)
    assert vec.dtype == np.float32


def test_encode_text_normalize_embeddings_flag():
    fake_st = _fake_st()
    emb = _embedder_with_st(fake_st)
    emb.encode_text("x")
    _, kwargs = fake_st.encode.call_args
    assert kwargs.get("normalize_embeddings") is True


def test_encode_text_show_progress_bar_off():
    fake_st = _fake_st()
    emb = _embedder_with_st(fake_st)
    emb.encode_text("x")
    _, kwargs = fake_st.encode.call_args
    assert kwargs.get("show_progress_bar") is False


def test_encode_text_model_loaded_once():
    """_load_mclip is called on the first encode_text, then cached."""
    emb = MClipEmbedder(cfg=None, device=None)
    load_calls = []

    def _mock_load(self):
        if self._st_model is not None:
            return
        load_calls.append(1)
        self._st_model = _fake_st()

    with patch.object(MClipEmbedder, "_load_mclip", _mock_load):
        emb.encode_text("first")
        emb.encode_text("second")

    assert len(load_calls) == 1


def test_encode_text_raises_on_missing_sentence_transformers():
    emb = MClipEmbedder(cfg=None, device=None)

    def _mock_load(self):
        raise RuntimeError("sentence-transformers not installed")

    with patch.object(MClipEmbedder, "_load_mclip", _mock_load):
        with pytest.raises(RuntimeError, match="sentence-transformers"):
            emb.encode_text("x")


def test_load_mclip_uses_correct_model_name():
    """SentenceTransformer is constructed with _ST_MODEL_NAME."""
    emb = MClipEmbedder(cfg=None, device=None)
    with patch("sentence_transformers.SentenceTransformer") as mock_cls:
        mock_cls.return_value = _fake_st()
        emb._load_mclip()
    mock_cls.assert_called_once_with(_ST_MODEL_NAME, device="cpu")


# ── image methods delegated to OpenClipEmbedder ───────────────────────────────

def test_encode_images_delegates_to_parent():
    emb = _embedder_with_st()
    fake_result = np.ones((2, 512), dtype="float32")
    with patch(
        "cinemateca.models.clip.openclip.OpenClipEmbedder.encode_images",
        return_value=fake_result,
    ) as mock_enc:
        result = emb.encode_images([Path("a.jpg"), Path("b.jpg")])
    mock_enc.assert_called_once()
    np.testing.assert_array_equal(result, fake_result)


def test_no_st_model_loaded_for_image_only_usage():
    """encode_image_single must not trigger M-CLIP text model loading."""
    emb = MClipEmbedder(cfg=None, device=None)
    fake_vec = np.ones(512, dtype="float32")
    with patch(
        "cinemateca.models.clip.openclip.OpenClipEmbedder.encode_image_single",
        return_value=fake_vec,
    ):
        emb.encode_image_single("img.jpg")
    assert emb._st_model is None


# ── registry dispatch ──────────────────────────────────────────────────────────

def _minimal_cfg(embedder_name: str) -> SimpleNamespace:
    return SimpleNamespace(
        models=SimpleNamespace(image_embedder=embedder_name),
        embeddings=SimpleNamespace(model="ViT-B-32", pretrained="openai", batch_size=16),
    )


def test_registry_returns_mclip_embedder():
    from cinemateca.models.registry import get_image_embedder

    embedder = get_image_embedder(_minimal_cfg("clip_mclip"))
    assert isinstance(embedder, MClipEmbedder)


def test_registry_clip_openclip_unchanged():
    from cinemateca.models.clip.openclip import OpenClipEmbedder
    from cinemateca.models.registry import get_image_embedder

    embedder = get_image_embedder(_minimal_cfg("clip_openclip"))
    assert isinstance(embedder, OpenClipEmbedder)
    assert not isinstance(embedder, MClipEmbedder)


def test_registry_unknown_name_raises():
    from cinemateca.models.registry import get_image_embedder

    with pytest.raises(ValueError, match="Unknown image_embedder"):
        get_image_embedder(_minimal_cfg("does_not_exist"))
