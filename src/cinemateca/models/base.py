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
from typing import Protocol, runtime_checkable

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


@runtime_checkable
class SceneDescriber(Protocol):
    """Generates natural-language metadata for a keyframe using a VLM."""

    def describe(self, image_path: str | Path) -> dict:
        """description/location/setting/time_of_day/num_people/objects/tags."""
        ...

    def describe_batch(
        self,
        keyframes_df: pd.DataFrame,
        existing_results: list[dict] | None = None,
        checkpoint_path: Path | None = None,
    ) -> list[dict]:
        """Process all rows; resume via existing_results."""
        ...


@runtime_checkable
class EnvironmentClassifier(Protocol):
    """Heuristic or model-based environment classification."""

    def classify(self, image_path: str | Path) -> dict:
        """Returns {"time_of_day","brightness_score","location","edge_density"}."""
        ...

    def classify_batch(self, image_paths: list[Path]) -> list[dict]:
        ...
