"""
cinemateca.models.registry
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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cinemateca.models.base import (
        AudioEmbedder,
        EnvironmentClassifier,
        FaceDetector,
        ImageEmbedder,
        ObjectDetector,
        SceneDescriber,
    )


def _name(cfg, attr: str) -> str:
    models = getattr(cfg, "models", None)
    if models is None:
        raise ValueError("config has no [models] section")
    val = getattr(models, attr, None)
    if not val:
        raise ValueError(f"models.{attr} is unset")
    return val


def get_image_embedder(cfg, device=None) -> "ImageEmbedder":
    name = _name(cfg, "image_embedder")
    if name == "clip_openclip":
        from cinemateca.models.clip.openclip import OpenClipEmbedder

        return OpenClipEmbedder(cfg, device)
    if name == "clip_mclip":
        from cinemateca.models.clip.mclip import MClipEmbedder

        return MClipEmbedder(cfg, device)
    if name == "siglip_multilingual":
        # M3 pre-flight Task 4.1: opt-in multilingual SigLIP backend.
        # Default stays clip_openclip; Task 4.2/4.3 own the flip + re-embed.
        from cinemateca.models.clip.siglip_multilingual import (
            SiglipMultilingualEmbedder,
        )

        return SiglipMultilingualEmbedder(cfg, device)
    raise ValueError(f"Unknown image_embedder: {name!r}")


def get_face_detector(cfg, device=None) -> "FaceDetector":
    name = _name(cfg, "face_detector")
    if name == "mtcnn_pytorch":
        from cinemateca.models.face.mtcnn import MTCNNFaceDetector

        return MTCNNFaceDetector(cfg, device)
    raise ValueError(f"Unknown face_detector: {name!r}")


def get_object_detector(cfg, device=None) -> "ObjectDetector":
    name = _name(cfg, "object_detector")
    if name == "yolov8":
        from cinemateca.models.objects.yolov8 import YOLOv8ObjectDetector

        return YOLOv8ObjectDetector(cfg, device)
    raise ValueError(f"Unknown object_detector: {name!r}")


def get_scene_describer(cfg, device=None) -> "SceneDescriber":
    name = _name(cfg, "scene_describer")
    # Order is load-bearing: moondream_transformers is the default backend.
    if name == "moondream_transformers":
        from cinemateca.models.describer.transformers_hf import (
            MoondreamTransformersDescriber,
        )

        return MoondreamTransformersDescriber(cfg, device)
    if name == "moondream_gguf":
        from cinemateca.models.describer.gguf import MoondreamGGUFDescriber

        return MoondreamGGUFDescriber(cfg, device)
    raise ValueError(f"Unknown scene_describer: {name!r}")


def get_environment_classifier(cfg, device=None) -> "EnvironmentClassifier":
    name = _name(cfg, "environment_classifier")
    if name == "opencv_heuristic":
        from cinemateca.models.environment.opencv_heuristic import (
            OpenCVEnvironmentClassifier,
        )

        return OpenCVEnvironmentClassifier(cfg, device)
    raise ValueError(f"Unknown environment_classifier: {name!r}")


def get_audio_embedder(cfg, device=None) -> "AudioEmbedder":
    name = _name(cfg, "audio_embedder")
    if name == "clap_hf":
        from cinemateca.models.audio.clap_hf import ClapHFEmbedder

        return ClapHFEmbedder(cfg, device)
    raise ValueError(f"Unknown audio_embedder: {name!r}")
