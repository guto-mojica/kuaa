"""Protocol + registry conformance (no model load, no GPU, hermetic)."""
from __future__ import annotations

import numpy as np


def test_base_protocols_exist_and_are_runtime_checkable():
    from cinemateca.models.base import (
        EnvironmentClassifier,
        FaceDetector,
        ImageEmbedder,
        ObjectDetector,
        SceneDescriber,
    )

    for proto in (
        ImageEmbedder, FaceDetector, ObjectDetector,
        SceneDescriber, EnvironmentClassifier,
    ):
        assert getattr(proto, "_is_runtime_protocol", False), proto


def test_protocols_isinstance_structural():
    """Structural isinstance checks — no real models, no GPU, numpy only."""
    from cinemateca.models.base import (
        EnvironmentClassifier,
        FaceDetector,
        ImageEmbedder,
        ObjectDetector,
        SceneDescriber,
    )

    # ------------------------------------------------------------------ #
    # ImageEmbedder
    # ------------------------------------------------------------------ #
    class _GoodEmbedder:
        def encode_images(self, image_paths):
            return np.zeros((len(image_paths), 4), dtype="float32")

        def encode_text(self, text):
            return np.zeros(4, dtype="float32")

        def encode_image_single(self, image_path):
            return np.zeros(4, dtype="float32")

    class _BadEmbedder:
        def encode_images(self, image_paths):
            return np.zeros((len(image_paths), 4), dtype="float32")

        # missing encode_text and encode_image_single

    assert isinstance(_GoodEmbedder(), ImageEmbedder) is True
    assert isinstance(_BadEmbedder(), ImageEmbedder) is False

    # expected method names are present on the protocol (3.11-compatible)
    _embedder_members = {m for m in vars(ImageEmbedder) if not m.startswith("_")}
    assert {"encode_images", "encode_text", "encode_image_single"}.issubset(_embedder_members)

    # ------------------------------------------------------------------ #
    # FaceDetector
    # ------------------------------------------------------------------ #
    class _GoodFace:
        def detect(self, image_path):
            return {"num_faces": 0, "faces": []}

        def detect_batch(self, image_paths):
            return [{"num_faces": 0, "faces": []} for _ in image_paths]

    class _BadFace:
        def detect(self, image_path):
            return {}
        # missing detect_batch

    assert isinstance(_GoodFace(), FaceDetector) is True
    assert isinstance(_BadFace(), FaceDetector) is False
    _face_members = {m for m in vars(FaceDetector) if not m.startswith("_")}
    assert {"detect", "detect_batch"}.issubset(_face_members)

    # ------------------------------------------------------------------ #
    # ObjectDetector
    # ------------------------------------------------------------------ #
    class _GoodObject:
        def detect(self, image_path):
            return {"num_objects": 0, "objects": [], "class_counts": {}}

        def detect_batch(self, image_paths):
            return [{"num_objects": 0, "objects": [], "class_counts": {}} for _ in image_paths]

    class _BadObject:
        def detect_batch(self, image_paths):
            return []
        # missing detect

    assert isinstance(_GoodObject(), ObjectDetector) is True
    assert isinstance(_BadObject(), ObjectDetector) is False
    _obj_members = {m for m in vars(ObjectDetector) if not m.startswith("_")}
    assert {"detect", "detect_batch"}.issubset(_obj_members)

    # ------------------------------------------------------------------ #
    # SceneDescriber
    # ------------------------------------------------------------------ #
    class _GoodDescriber:
        def describe(self, image_path):
            return {"description": "", "tags": []}

        def describe_batch(self, keyframes_df, existing_results=None, checkpoint_path=None):
            return [{"description": ""} for _ in range(len(keyframes_df))]

    class _BadDescriber:
        def describe(self, image_path):
            return {}
        # missing describe_batch

    assert isinstance(_GoodDescriber(), SceneDescriber) is True
    assert isinstance(_BadDescriber(), SceneDescriber) is False
    _desc_members = {m for m in vars(SceneDescriber) if not m.startswith("_")}
    assert {"describe", "describe_batch"}.issubset(_desc_members)

    # ------------------------------------------------------------------ #
    # EnvironmentClassifier
    # ------------------------------------------------------------------ #
    class _GoodClassifier:
        def classify(self, image_path):
            return {"time_of_day": "day", "brightness_score": 0.5,
                    "location": "indoor", "edge_density": 0.1}

        def classify_batch(self, image_paths):
            return [self.classify(p) for p in image_paths]

    class _BadClassifier:
        def classify_batch(self, image_paths):
            return []
        # missing classify

    assert isinstance(_GoodClassifier(), EnvironmentClassifier) is True
    assert isinstance(_BadClassifier(), EnvironmentClassifier) is False
    _env_members = {m for m in vars(EnvironmentClassifier) if not m.startswith("_")}

    assert {"classify", "classify_batch"}.issubset(_env_members)


def test_detector_backends_conform():
    from cinemateca.models.base import (
        EnvironmentClassifier,
        FaceDetector,
        ObjectDetector,
    )
    from cinemateca.models.environment.opencv_heuristic import (
        OpenCVEnvironmentClassifier,
    )
    from cinemateca.models.face.mtcnn import MTCNNFaceDetector
    from cinemateca.models.objects.yolov8 import YOLOv8ObjectDetector

    assert isinstance(MTCNNFaceDetector(), FaceDetector)
    assert isinstance(YOLOv8ObjectDetector(), ObjectDetector)
    assert isinstance(OpenCVEnvironmentClassifier(), EnvironmentClassifier)


def test_visual_analyzer_injection():
    """VisualAnalyzer accepts injected backends and exposes them."""
    from cinemateca.models.environment.opencv_heuristic import (
        OpenCVEnvironmentClassifier,
    )
    from cinemateca.models.face.mtcnn import MTCNNFaceDetector
    from cinemateca.models.objects.yolov8 import YOLOv8ObjectDetector
    from cinemateca.visual_analyzer import VisualAnalyzer

    fd, od, ec = (
        MTCNNFaceDetector(),
        YOLOv8ObjectDetector(),
        OpenCVEnvironmentClassifier(),
    )
    va = VisualAnalyzer(face_detector=fd, object_detector=od, env_classifier=ec)
    assert va.face_detector is fd
    assert va.object_detector is od
    assert va.env_classifier is ec


def test_openclip_embedder_conforms():
    from cinemateca.models.base import ImageEmbedder
    from cinemateca.models.clip.openclip import OpenClipEmbedder

    assert isinstance(OpenClipEmbedder(), ImageEmbedder)


def test_by_image_uses_encode_image_single(monkeypatch, tmp_path):
    """by_image must call embedder.encode_image_single, not embedder privates."""
    import numpy as np
    import pandas as pd

    from cinemateca.embeddings import SemanticSearch

    calls = {}

    class FakeEmbedder:
        def encode_image_single(self, image_path):
            calls["path"] = str(image_path)
            return np.array([1.0, 0.0], dtype="float32")

    emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    df = pd.DataFrame({"filepath": ["a.jpg", "b.jpg"], "scene_id": [1, 2]})
    s = SemanticSearch(emb, df, FakeEmbedder())
    out = s.by_image("query.jpg", top_k=2, exclude_self=False)
    assert calls["path"] == "query.jpg"
    assert list(out["scene_id"]) == [1, 2]
