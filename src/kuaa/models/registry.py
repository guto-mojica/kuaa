"""
kuaa.models.registry
~~~~~~~~~~~~~~~~~~~~~~~~~~
Factory functions that construct model backends from config.

Each ``get_*`` function reads ``cfg.models.<role>`` to select a backend
name, then constructs and returns the corresponding concrete backend.
The pipeline imports only from here — never from a concrete backend module
— so swapping backends requires only a config change.

Device is passed explicitly by the caller; this module never reads it
from cfg.

Return-type annotations use the Protocol types from ``base.py`` — they
are the public contract; callers must not rely on concrete backend types.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kuaa.config import Settings
    from kuaa.models.base import (
        EnvironmentClassifier,
        FaceDetector,
        ImageEmbedder,
        ObjectDetector,
        SceneDescriber,
    )
    from kuaa.models.manifest import ModelCard


_image_embedder_cache: dict[tuple[str, str | None], Any] = {}
# Guards the check-then-set below. Without it, two threads racing the
# same cache miss (e.g. the app-startup warm-up thread and a real
# request landing in the same window) both pass the "not cached" check,
# both construct+load a full model, and the second write silently
# discards the first — doubling load cost/memory exactly when startup
# should be lightest. A held lock across the (possibly ~10s) load also
# makes any thread arriving mid-load simply wait for the same cached
# instance instead of duplicating it.
_image_embedder_lock = threading.Lock()


def _name(cfg: Settings, attr: str) -> str:
    models = getattr(cfg, "models", None)
    if models is None:
        raise ValueError("config has no [models] section")
    val = getattr(models, attr, None)
    if not val:
        raise ValueError(f"models.{attr} is unset")
    return val


def _device_key(device) -> str | None:
    return None if device is None else str(device)


def reset_caches() -> None:
    """Drop cached embedder singletons.

    The image embedder factory memoises by (backend_name, device) so
    query-time encoding doesn't reconstruct weights every call. Tests that
    monkey-patch the underlying backend classes should call this in setup
    so the cache returns a fresh build.
    """
    with _image_embedder_lock:
        _image_embedder_cache.clear()


def get_image_embedder(cfg: Settings, device=None) -> ImageEmbedder:
    """Return the configured image-embedding backend.

    Provenance: see ModelCard via ``model_card(cfg, "image_embedder")``.

    Thread-safe: the cache check, construction, and store all happen under
    ``_image_embedder_lock``, so two callers racing the same cache miss
    (e.g. an app-startup warm-up thread and a real request) block on each
    other instead of each constructing — and loading — their own instance.
    """
    name = _name(cfg, "image_embedder")
    key = (name, _device_key(device))
    with _image_embedder_lock:
        cached = _image_embedder_cache.get(key)
        if cached is not None:
            return cached
        if name == "clip_openclip":
            from kuaa.models.clip.openclip import OpenClipEmbedder

            instance: ImageEmbedder = OpenClipEmbedder(cfg, device)
        elif name == "clip_mclip":
            from kuaa.models.clip.mclip import MClipEmbedder

            instance = MClipEmbedder(cfg, device)
        elif name == "siglip_multilingual":
            from kuaa.models.clip.siglip_multilingual import (
                SiglipMultilingualEmbedder,
            )

            instance = SiglipMultilingualEmbedder(cfg, device)
        else:
            raise ValueError(f"Unknown image_embedder: {name!r}")
        _image_embedder_cache[key] = instance
        return instance


def get_face_detector(cfg: Settings, device=None) -> FaceDetector:
    """Return the configured face-detector backend.

    Provenance: see ModelCard via ``model_card(cfg, "face_detector")``.
    """
    name = _name(cfg, "face_detector")
    if name == "mtcnn_pytorch":
        from kuaa.models.face.mtcnn import MTCNNFaceDetector

        return MTCNNFaceDetector(cfg, device)
    raise ValueError(f"Unknown face_detector: {name!r}")


def get_object_detector(cfg: Settings, device=None) -> ObjectDetector:
    """Return the configured object-detector backend.

    Provenance: see ModelCard via ``model_card(cfg, "object_detector")``.
    """
    name = _name(cfg, "object_detector")
    if name == "yolov8":
        from kuaa.models.objects.yolov8 import YOLOv8ObjectDetector

        return YOLOv8ObjectDetector(cfg, device)
    raise ValueError(f"Unknown object_detector: {name!r}")


def get_scene_describer(cfg: Settings, device=None) -> SceneDescriber:
    """Return the configured scene-describer (VLM) backend.

    Provenance: see ModelCard via ``model_card(cfg, "scene_describer")``.
    """
    name = _name(cfg, "scene_describer")
    # Order is load-bearing: moondream_transformers is the default backend.
    if name == "moondream_transformers":
        from kuaa.models.describer.transformers_hf import (
            MoondreamTransformersDescriber,
        )

        return MoondreamTransformersDescriber(cfg, device)
    if name == "moondream_gguf":
        from kuaa.models.describer.gguf import MoondreamGGUFDescriber

        return MoondreamGGUFDescriber(cfg, device)
    raise ValueError(f"Unknown scene_describer: {name!r}")


def get_environment_classifier(cfg: Settings, device=None) -> EnvironmentClassifier:
    """Return the configured environment-classifier backend.

    Provenance: see ModelCard via ``model_card(cfg, "environment_classifier")``.
    """
    name = _name(cfg, "environment_classifier")
    if name == "opencv_heuristic":
        from kuaa.models.environment.opencv_heuristic import (
            OpenCVEnvironmentClassifier,
        )

        return OpenCVEnvironmentClassifier(cfg, device)
    raise ValueError(f"Unknown environment_classifier: {name!r}")


# ---------------------------------------------------------------------------
# Config-aware manifest accessor (F6)
# ---------------------------------------------------------------------------

#: Model roles served by ``settings.models.*`` selectors.
_MODELS_ROLES = frozenset(
    {
        "image_embedder",
        "face_detector",
        "object_detector",
        "scene_describer",
        "environment_classifier",
    }
)


def model_card(settings: Settings, role: str) -> ModelCard:
    """Return the :class:`~kuaa.models.manifest.ModelCard` for *role*.

    Resolves the *active* backend from *settings* so the returned card
    always matches the configured backend — not a role-level default.

    For the five roles with a ``settings.models.*`` selector
    (``image_embedder``, ``face_detector``, ``object_detector``,
    ``scene_describer``, ``environment_classifier``), the backend id is
    read from ``settings.models`` and used as the
    :data:`~kuaa.models.manifest.CARDS` key.

    ``"reranker"`` has no ``settings.models`` selector (it is configured
    under ``settings.retrieval``); this function returns the single
    ``bge_reranker_v2_m3`` card directly.

    Raises ``KeyError`` for unknown roles.

    Used by provenance-aware code (WS-1 C10) and the docs renderers
    (WS-6 D4/D9) so model identity has a single source of truth.
    """
    from kuaa.models.manifest import get_card

    if role in _MODELS_ROLES:
        backend: str = getattr(settings.models, role)
        return get_card(backend)
    if role == "reranker":
        return get_card("bge_reranker_v2_m3")
    raise KeyError(role)
