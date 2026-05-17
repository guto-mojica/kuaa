"""
cinemateca.models.registry
~~~~~~~~~~~~~~~~~~~~~~~~~~
Reads cfg.models.* and returns concrete backends. The pipeline imports
from here, never from a concrete backend module.
"""
from __future__ import annotations


def _name(cfg, attr: str) -> str:
    models = getattr(cfg, "models", None)
    if models is None:
        raise ValueError("config has no [models] section")
    val = getattr(models, attr, None)
    if not val:
        raise ValueError(f"models.{attr} is unset")
    return val


def get_image_embedder(cfg):
    name = _name(cfg, "image_embedder")
    if name == "clip_openclip":
        from cinemateca.models.clip.openclip import OpenClipEmbedder
        return OpenClipEmbedder(cfg, getattr(cfg, "_device", None))
    raise ValueError(f"Unknown image_embedder: {name!r}")


def get_face_detector(cfg):
    name = _name(cfg, "face_detector")
    if name == "mtcnn_pytorch":
        from cinemateca.models.face.mtcnn import MTCNNFaceDetector
        return MTCNNFaceDetector(cfg, getattr(cfg, "_device", None))
    raise ValueError(f"Unknown face_detector: {name!r}")


def get_object_detector(cfg):
    name = _name(cfg, "object_detector")
    if name == "yolov8":
        from cinemateca.models.objects.yolov8 import YOLOv8ObjectDetector
        return YOLOv8ObjectDetector(cfg, getattr(cfg, "_device", None))
    raise ValueError(f"Unknown object_detector: {name!r}")


def get_scene_describer(cfg):
    name = _name(cfg, "scene_describer")
    if name == "moondream_gguf":
        from cinemateca.models.describer.gguf import MoondreamGGUFDescriber
        return MoondreamGGUFDescriber(cfg, getattr(cfg, "_device", None))
    raise ValueError(f"Unknown scene_describer: {name!r}")


def get_environment_classifier(cfg):
    name = _name(cfg, "environment_classifier")
    if name == "opencv_heuristic":
        from cinemateca.models.environment.opencv_heuristic import (
            OpenCVEnvironmentClassifier,
        )
        return OpenCVEnvironmentClassifier(cfg)
    raise ValueError(f"Unknown environment_classifier: {name!r}")
