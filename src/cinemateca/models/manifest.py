"""Model provenance manifest (F6).

A :class:`ModelCard` per model backend: identity, revision, vector dim /
I-O shape, license, download hint, and an optional checksum.  ``CARDS``
is keyed by **backend id** (the same strings used in ``models.*`` config
selectors), so the active card for a role is simply
``CARDS[settings.models.<role>]``.  Two backends that serve the same role
(``clip_openclip`` / ``siglip_multilingual`` and
``moondream_transformers`` / ``moondream_gguf``) are both present so the
manifest always reflects the *configured* backend, not a role-level
default.

The registry exposes :func:`cinemateca.models.registry.model_card` as the
config-aware entry point; docs (WS-6 D4/D9) render
``MODEL_INVENTORY`` / ``LICENSES`` from these cards so there is no drift
between code and documentation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelCard:
    """Provenance for one model backend.

    ``dim`` is the embedding dimension where applicable (else ``None``);
    ``output_shape`` documents the producer contract (e.g.
    ``"(N, D) float32 L2-normalised"``).  ``revision`` pins the exact
    HF commit / tag that was validated; ``checksum`` is the known weights
    hash when pinned, else ``None`` (weights download on first use from
    the hub).

    Invariant: ``card.backend == CARDS`` key for every entry.
    """

    role: str
    backend: str
    model_id: str
    license: str
    revision: str | None = None
    dim: int | None = None
    input_shape: str | None = None
    output_shape: str | None = None
    download_hint: str | None = None
    checksum: str | None = None


# ---------------------------------------------------------------------------
# Active + alternate backends — keyed by backend id.
# The ``models.*`` config selectors (``clip_openclip``, ``siglip_multilingual``,
# ``moondream_transformers``, ``moondream_gguf``, …) are exactly these keys.
# ---------------------------------------------------------------------------

CARDS: dict[str, ModelCard] = {
    # ── image_embedder: default backend ─────────────────────────────────────
    "siglip_multilingual": ModelCard(
        role="image_embedder",
        backend="siglip_multilingual",
        model_id="google/siglip2-large-patch16-256",
        license="Apache-2.0",
        dim=1024,
        output_shape="(N, 1024) float32 L2-normalised; shared text+image space",
        download_hint="HF hub on first encode (~1.5 GB)",
    ),
    # ── image_embedder: alternate backend ────────────────────────────────────
    "clip_openclip": ModelCard(
        role="image_embedder",
        backend="clip_openclip",
        model_id="ViT-B-32",
        license="MIT (OpenCLIP); OpenAI checkpoint terms apply for 'openai' pretrained",
        dim=512,
        output_shape="(N, 512) float32 L2-normalised; shared text+image space",
        download_hint="open-clip-torch downloads ViT-B-32/openai weights on first encode",
    ),
    # ── scene_describer: default backend ─────────────────────────────────────
    "moondream_transformers": ModelCard(
        role="scene_describer",
        backend="moondream_transformers",
        model_id="vikhyatk/moondream2",
        revision="2025-01-09",
        license="Apache-2.0",
        download_hint="HF hub on first describe (~2 GB); transformers>=4.44,<5",
    ),
    # ── scene_describer: alternate backend (GGUF / llama-cpp-python) ─────────
    "moondream_gguf": ModelCard(
        role="scene_describer",
        backend="moondream_gguf",
        model_id="vikhyatk/moondream2",
        revision="2025-01-09",
        license="Apache-2.0",
        download_hint=(
            "HF hub: moondream2-text-model-f16.gguf + moondream2-mmproj-f16.gguf "
            "(~2 GB); GPU requires llama-cpp-python built from source with CUDA "
            "(see docs/GPU_LLAMA_CPP_CUDA_BUILD.md)"
        ),
    ),
    # ── object_detector ──────────────────────────────────────────────────────
    "yolov8": ModelCard(
        role="object_detector",
        backend="yolov8",
        model_id="yolov8n.pt",
        license="AGPL-3.0 (Ultralytics; Enterprise license required for proprietary use)",
        download_hint="ultralytics fetches weights on first detect",
    ),
    # ── face_detector ────────────────────────────────────────────────────────
    "mtcnn_pytorch": ModelCard(
        role="face_detector",
        backend="mtcnn_pytorch",
        model_id="facenet-pytorch:MTCNN",
        license="MIT",
        output_shape='{"num_faces": int, "faces": [...]}',
    ),
    # ── environment_classifier ────────────────────────────────────────────────
    "opencv_heuristic": ModelCard(
        role="environment_classifier",
        backend="opencv_heuristic",
        model_id="local:opencv_heuristic",
        license="project (local heuristic, no weights)",
        output_shape='{"time_of_day", "brightness_score", "location", "edge_density"}',
    ),
    # ── reranker ─────────────────────────────────────────────────────────────
    "bge_reranker_v2_m3": ModelCard(
        role="reranker",
        backend="bge_reranker_v2_m3",
        model_id="BAAI/bge-reranker-v2-m3",
        license="Apache-2.0",
        download_hint="HF hub on first rerank",
    ),
}

# Invariant: every card's .backend must equal its CARDS key.
assert all(
    card.backend == key for key, card in CARDS.items()
), "CARDS invariant violated: card.backend must equal its key"


def get_card(backend: str) -> ModelCard:
    """Return the :class:`ModelCard` for ``backend`` or raise ``KeyError``."""
    return CARDS[backend]


__all__ = ["CARDS", "ModelCard", "get_card"]
