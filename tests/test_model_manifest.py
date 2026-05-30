"""Model manifest (F6) — backend-keyed ModelCard registry.

Spec §17 override: CARDS is keyed by backend id (not role), so the
active card for a role is CARDS[settings.models.<role>].  Nine backends
are registered: two image backends (default siglip_multilingual + alt
clip_openclip) and two describer backends (default
moondream_transformers + alt moondream_gguf) plus one backend per other
role.
"""

from __future__ import annotations

import pytest

from cinemateca.models.manifest import CARDS, ModelCard

EXPECTED_BACKENDS = {
    "siglip_multilingual",
    "clip_openclip",
    "clap_hf",
    "moondream_transformers",
    "moondream_gguf",
    "yolov8",
    "mtcnn_pytorch",
    "opencv_heuristic",
    "bge_reranker_v2_m3",
}


def test_all_expected_backend_keys_present():
    assert set(CARDS) == EXPECTED_BACKENDS


@pytest.mark.parametrize("backend", sorted(EXPECTED_BACKENDS))
def test_each_card_is_wellformed(backend):
    card = CARDS[backend]
    assert isinstance(card, ModelCard)
    # Invariant: card.backend matches its key in CARDS.
    assert card.backend == backend
    assert card.role  # non-empty role string
    assert card.model_id  # non-empty model identifier
    assert card.license  # license stated


def test_siglip_multilingual_dim():
    assert CARDS["siglip_multilingual"].dim == 1024


def test_clap_hf_dim():
    assert CARDS["clap_hf"].dim == 512


def test_moondream_transformers_revision():
    assert CARDS["moondream_transformers"].revision == "2025-01-09"


def test_yolov8_agpl_license():
    assert "AGPL" in CARDS["yolov8"].license


def test_both_image_backends_share_role():
    assert CARDS["siglip_multilingual"].role == "image_embedder"
    assert CARDS["clip_openclip"].role == "image_embedder"


def test_both_describer_backends_share_role():
    assert CARDS["moondream_transformers"].role == "scene_describer"
    assert CARDS["moondream_gguf"].role == "scene_describer"


def test_clip_openclip_dim():
    assert CARDS["clip_openclip"].dim == 512


def test_moondream_gguf_revision_matches_transformers():
    """Both describer backends pin the same validated revision."""
    assert CARDS["moondream_gguf"].revision == CARDS["moondream_transformers"].revision
