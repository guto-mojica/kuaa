"""Pre-processing: scene-cut review + manual boundary editing.

HTTP-agnostic logic for the Pre-processing surface. Reads/writes the
authoritative ``scene_cuts.json`` cut set (see :mod:`kuaa.scene_detector`),
stages split/merge edits cheaply in ``pending_edits.json``, applies them with
a partial (rename-what's-unchanged, re-extract-what's-touched) rebuild, and
builds the filmstrip view model the template renders.
"""

from __future__ import annotations

from kuaa.preprocess.service import (
    CutEditError,
    PendingEdits,
    PendingOp,
    apply_pending,
    build_filmstrip,
    discard_pending,
    has_pending,
    load_or_derive_cutset,
    stage_merge,
    stage_split,
    toggle_pending,
)

__all__ = [
    "CutEditError",
    "PendingEdits",
    "PendingOp",
    "apply_pending",
    "build_filmstrip",
    "discard_pending",
    "has_pending",
    "load_or_derive_cutset",
    "stage_merge",
    "stage_split",
    "toggle_pending",
]
