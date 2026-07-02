"""Pre-processing: scene-cut review + manual boundary editing.

HTTP-agnostic logic for the Pre-processing surface. Reads/writes the
authoritative ``scene_cuts.json`` cut set (see :mod:`kuaa.scene_detector`),
rebuilds keyframes + ``keyframes_metadata.json`` from it, and builds the
filmstrip view model the template renders.
"""

from __future__ import annotations

from kuaa.preprocess.service import (
    CutEditError,
    build_filmstrip,
    load_or_derive_cutset,
    merge_at,
    split_scene,
)

__all__ = [
    "CutEditError",
    "build_filmstrip",
    "load_or_derive_cutset",
    "merge_at",
    "split_scene",
]
