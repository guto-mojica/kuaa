"""Model manifest (F6) — backend-keyed ModelCard registry.

Spec §17 override: CARDS is keyed by backend id (not role), so the
active card for a role is CARDS[settings.models.<role>].  Nine backends
are registered: two image backends (default siglip_multilingual + alt
clip_openclip) and two describer backends (default
moondream_transformers + alt moondream_gguf) plus one backend per other
role.
"""

from __future__ import annotations

import importlib

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


def _shipped_backend_cards() -> dict[str, ModelCard]:
    """Collect the ``CARD`` reference declared by every shipped backend.

    Each concrete backend under ``cinemateca.models.**`` (plus the reranker,
    which lives in ``cinemateca.search.rerank`` because it has no
    ``models.*`` selector) declares a ``CARD`` link to its manifest entry.
    Returns a ``{label: ModelCard}`` map. The label is the backend's import
    site, used only for assertion messages.

    Note: ``cinemateca.search.rerank`` is loaded via :func:`importlib.import_module`
    because the ``cinemateca.search`` package re-exports the ``rerank``
    *function* under that same name, which shadows the submodule on a plain
    ``import cinemateca.search.rerank as ...``.
    """
    from cinemateca.models.audio.clap_hf import ClapHFEmbedder
    from cinemateca.models.clip.openclip import OpenClipEmbedder
    from cinemateca.models.clip.siglip_multilingual import SiglipMultilingualEmbedder
    from cinemateca.models.describer.gguf import MoondreamGGUFDescriber
    from cinemateca.models.describer.transformers_hf import MoondreamTransformersDescriber
    from cinemateca.models.environment.opencv_heuristic import OpenCVEnvironmentClassifier
    from cinemateca.models.face.mtcnn import MTCNNFaceDetector
    from cinemateca.models.objects.yolov8 import YOLOv8ObjectDetector

    rerank_mod = importlib.import_module("cinemateca.search.rerank")

    return {
        "OpenClipEmbedder": OpenClipEmbedder.CARD,
        "SiglipMultilingualEmbedder": SiglipMultilingualEmbedder.CARD,
        "ClapHFEmbedder": ClapHFEmbedder.CARD,
        "MoondreamTransformersDescriber": MoondreamTransformersDescriber.CARD,
        "MoondreamGGUFDescriber": MoondreamGGUFDescriber.CARD,
        "YOLOv8ObjectDetector": YOLOv8ObjectDetector.CARD,
        "MTCNNFaceDetector": MTCNNFaceDetector.CARD,
        "OpenCVEnvironmentClassifier": OpenCVEnvironmentClassifier.CARD,
        "rerank (module)": rerank_mod.CARD,
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


# ── C10: every shipped backend references its ModelCard ───────────────────────


def test_every_shipped_backend_references_a_real_card():
    """Each backend's ``CARD`` resolves to a real, canonical ``CARDS`` entry.

    Guards against a backend drifting from the manifest (e.g. a copy-pasted
    ``get_card("...")`` with a stale id, or a subclass silently inheriting the
    wrong parent ``CARD``).
    """
    cards = _shipped_backend_cards()
    for label, card in cards.items():
        assert isinstance(card, ModelCard), f"{label}.CARD is not a ModelCard: {card!r}"
        assert card.backend in CARDS, f"{label}.CARD.backend {card.backend!r} not in CARDS"
        # Identity, not just equality: the backend must point at the canonical
        # manifest object, so docs/provenance renderers and the backend agree.
        assert (
            CARDS[card.backend] is card
        ), f"{label}.CARD is not the canonical CARDS[{card.backend!r}] object"


def test_shipped_backends_cover_every_card_exactly_once():
    """The carded backends partition ``CARDS``: full coverage, no duplicates.

    If a new card is added to the manifest without a backend linking to it
    (or vice versa), this fails — keeping manifest and backends in lockstep.
    """
    referenced = [card.backend for card in _shipped_backend_cards().values()]
    # No backend id is claimed twice (each card maps to exactly one backend).
    assert len(referenced) == len(set(referenced)), f"duplicate backend links: {referenced}"
    assert set(referenced) == set(
        CARDS
    ), f"backend↔manifest mismatch (symmetric diff): {set(referenced) ^ set(CARDS)}"


def test_mclip_has_no_card_does_not_inherit_openclip():
    """M-CLIP is an unshipped fallback: ``CARD`` is explicitly ``None``.

    It subclasses :class:`OpenClipEmbedder`, so without an explicit override it
    would inherit ``clip_openclip`` and be mislabeled (it swaps in a different
    text encoder). The explicit ``None`` is the guard; this test pins it.
    """
    from cinemateca.models.clip.mclip import MClipEmbedder

    assert MClipEmbedder.CARD is None
