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
