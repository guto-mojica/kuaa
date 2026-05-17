"""Protocol + registry conformance (no model load, no GPU, hermetic)."""
from __future__ import annotations


def test_base_protocols_exist_and_are_runtime_checkable():
    from typing import get_type_hints  # noqa: F401

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
