"""Registry → config-aware ModelCard accessor (F6).

model_card(settings, role) resolves the active backend for a role from
settings.models.* and returns the corresponding ModelCard from the
manifest.  The 'reranker' role has no settings.models selector
(configured under retrieval.*) so it always returns the single
bge_reranker_v2_m3 card.
"""

from __future__ import annotations

import pytest

from cinemateca.config import load_config
from cinemateca.models.manifest import ModelCard
from cinemateca.models.registry import model_card

_ALL_ROLES = (
    "image_embedder",
    "audio_embedder",
    "scene_describer",
    "object_detector",
    "face_detector",
    "environment_classifier",
    "transcriber",
    "reranker",
)


def test_model_card_returns_card_for_each_active_role():
    cfg = load_config(ensure_dirs=False)
    for role in _ALL_ROLES:
        card = model_card(cfg, role)
        assert isinstance(card, ModelCard)
        assert card.role == role


def test_model_card_active_backend_resolution():
    """The returned card's backend matches what the config selects."""
    cfg = load_config(ensure_dirs=False)
    card = model_card(cfg, "image_embedder")
    assert card.backend == cfg.models.image_embedder


def test_model_card_reranker_is_config_independent():
    """reranker has no models.* selector — always returns bge_reranker_v2_m3."""
    cfg = load_config(ensure_dirs=False)
    card = model_card(cfg, "reranker")
    assert card.backend == "bge_reranker_v2_m3"
    assert card.role == "reranker"


def test_model_card_unknown_role_raises():
    cfg = load_config(ensure_dirs=False)
    with pytest.raises(KeyError):
        model_card(cfg, "unknown_role")
