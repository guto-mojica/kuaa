"""
cinemateca.models.base
~~~~~~~~~~~~~~~~~~~~~~
Protocol definitions for every model role in the pipeline.

Concrete backends (openclip, mtcnn, yolov8, gguf, …) implement these
interfaces. The pipeline imports only from here / the registry — never
from a concrete backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, TypedDict, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class ImageEmbedder(Protocol):
    """Encodes images into L2-normalised float32 vectors."""

    def encode_images(self, image_paths: list[Path]) -> np.ndarray:
        """Returns (N, D) float32 array, L2-normalised."""
        ...

    def encode_text(self, text: str) -> np.ndarray:
        """Returns (D,) float32 vector, L2-normalised (shared CLIP space)."""
        ...

    def encode_image_single(self, image_path: str | Path) -> np.ndarray:
        """Returns (D,) float32 vector for one image (image search)."""
        ...

    def save(
        self,
        embeddings: np.ndarray,
        keyframes_df: pd.DataFrame,
        output_dir: str | Path,
        embeddings_filename: str = ...,
        mapping_filename: str = ...,
    ) -> tuple[Path, Path]:
        """Persist embeddings + mapping; return (emb_path, map_path)."""
        ...


@runtime_checkable
class FaceDetector(Protocol):
    """Detects human faces in keyframe images."""

    def detect(self, image_path: str | Path) -> dict:
        """Returns {"num_faces": int, "faces": [...]}."""
        ...

    def detect_batch(self, image_paths: list[Path]) -> list[dict]:
        """One result dict per path, same order as input."""
        ...


@runtime_checkable
class ObjectDetector(Protocol):
    """Detects objects in keyframe images."""

    def detect(self, image_path: str | Path) -> dict:
        """Returns {"num_objects": int, "objects": [...], "class_counts": {...}}."""
        ...

    def detect_batch(self, image_paths: list[Path]) -> list[dict]:
        """One result dict per path, same order as input."""
        ...


class SceneDescriptionRecord(TypedDict, total=False):
    """One scene-description checkpoint row (resume shape for describe_batch).

    ``scene_id`` + ``description`` are always present in a successful row;
    the remaining keys are the structured Moondream outputs or scene-level
    metadata fields populated by :func:`~cinemateca.models.describer._common.build_metadata`.
    Error rows omit ``description`` and carry ``error`` instead.
    """

    scene_id: int
    description: str
    location: str
    setting: str
    time_of_day: str
    num_people: int
    people_action: str
    objects: list[str]
    tags: list[str]
    # scene-level provenance fields (from build_metadata / pipeline row)
    keyframe_id: str
    keyframe_path: str
    start_time_s: float | None
    end_time_s: float | None
    duration_s: float | None
    # error path (mutually exclusive with description)
    error: str


@runtime_checkable
class SceneDescriber(Protocol):
    """Generates natural-language metadata for a keyframe using a VLM."""

    def describe(self, image_path: str | Path) -> dict:
        """description/location/setting/time_of_day/num_people/objects/tags."""
        ...

    def describe_batch(
        self,
        keyframes_df: pd.DataFrame,
        existing_results: list[SceneDescriptionRecord] | None = None,
        checkpoint_path: Path | None = None,
    ) -> list[SceneDescriptionRecord]:
        """Describe all rows; resume from ``existing_results`` if provided."""
        ...

    @staticmethod
    def build_tag_index(results: list[SceneDescriptionRecord]) -> dict[str, list[str]]:
        """Build a tag → [scene_id] inverted index from description results."""
        ...

    def save(
        self,
        results: list[SceneDescriptionRecord],
        tag_index: dict[str, Any],
        output_dir: str | Path,
    ) -> None:
        """Persist description results and tag index as JSON files under output_dir."""
        ...


@runtime_checkable
class EnvironmentClassifier(Protocol):
    """Heuristic or model-based environment classification.

    Note: the bundled OpenCV backend is an approximate heuristic based on
    simple brightness and edge-density metrics. For production use, train
    an institution-specific classifier on the archive's own collection.
    """

    def classify(self, image_path: str | Path) -> dict:
        """Returns {"time_of_day","brightness_score","location","edge_density"}."""
        ...

    def classify_batch(self, image_paths: list[Path]) -> list[dict]: ...
