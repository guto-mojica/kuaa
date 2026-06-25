"""Human annotation correction statistics for evaluation reports."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kuaa.annotations import FILENAME as MANUAL_ANNOTATIONS_FILENAME
from kuaa.errors import ArtefactError
from kuaa.scene_ids import scene_id_key


class AnnotationStatsError(ArtefactError):
    """Raised when annotation metadata cannot be read."""

    default_code = "eval.annotation_unreadable"


@dataclass(frozen=True)
class AnnotationStats:
    """Aggregate generated-vs-manual tag comparison."""

    scenes_with_manual_annotations: int
    generated_tag_assignments_on_annotated_scenes: int
    manual_tag_assignments: int
    accepted_ai_tag_assignments: int
    human_added_tag_assignments: int
    human_removed_generated_tag_assignments: int
    correction_events: int
    correction_rate: float
    scenes_with_human_added_tags: int
    scenes_with_removed_generated_tags: int

    def to_dict(self) -> dict[str, int | float]:
        return {
            "scenes_with_manual_annotations": self.scenes_with_manual_annotations,
            "generated_tag_assignments_on_annotated_scenes": (
                self.generated_tag_assignments_on_annotated_scenes
            ),
            "manual_tag_assignments": self.manual_tag_assignments,
            "accepted_ai_tag_assignments": self.accepted_ai_tag_assignments,
            "human_added_tag_assignments": self.human_added_tag_assignments,
            "human_removed_generated_tag_assignments": (
                self.human_removed_generated_tag_assignments
            ),
            "correction_events": self.correction_events,
            "correction_rate": self.correction_rate,
            "scenes_with_human_added_tags": self.scenes_with_human_added_tags,
            "scenes_with_removed_generated_tags": self.scenes_with_removed_generated_tags,
        }


def normalize_tag(value: Any) -> str:
    """Normalize tags the same way manual annotation save does."""

    return str(value).strip().lower().replace(" ", "-")


def tag_index_to_scene_tags(
    tag_index: Mapping[str, Iterable[Any]] | None,
) -> dict[str, set[str]]:
    """Convert `{tag: [scene_ids]}` into `{scene_id: {tags}}`."""

    by_scene: dict[str, set[str]] = {}
    if not tag_index:
        return by_scene
    for raw_tag, ids in tag_index.items():
        tag = normalize_tag(raw_tag)
        if not tag:
            continue
        for raw_scene_id in ids:
            by_scene.setdefault(scene_id_key(raw_scene_id), set()).add(tag)
    return by_scene


def normalize_manual_annotations(
    annotations: Mapping[str, Iterable[Any]] | None,
) -> dict[str, set[str]]:
    """Normalize manual `{scene_id: [tags]}` annotations."""

    normalized: dict[str, set[str]] = {}
    if not annotations:
        return normalized
    for raw_scene_id, tags in annotations.items():
        scene_tags = {normalize_tag(t) for t in tags}
        scene_tags.discard("")
        normalized[scene_id_key(raw_scene_id)] = scene_tags
    return normalized


def compute_annotation_stats(
    generated_tag_index: Mapping[str, Iterable[Any]] | None,
    manual_annotations: Mapping[str, Iterable[Any]] | None,
) -> AnnotationStats:
    """Compare generated tag index with manual scene annotations."""

    generated_by_scene = tag_index_to_scene_tags(generated_tag_index)
    manual_by_scene = normalize_manual_annotations(manual_annotations)

    accepted = 0
    added = 0
    removed = 0
    generated_total = 0
    manual_total = 0
    scenes_added = 0
    scenes_removed = 0

    for scene_id, manual_tags in manual_by_scene.items():
        generated_tags = generated_by_scene.get(scene_id, set())
        accepted_tags = generated_tags & manual_tags
        added_tags = manual_tags - generated_tags
        removed_tags = generated_tags - manual_tags

        accepted += len(accepted_tags)
        added += len(added_tags)
        removed += len(removed_tags)
        generated_total += len(generated_tags)
        manual_total += len(manual_tags)
        if added_tags:
            scenes_added += 1
        if removed_tags:
            scenes_removed += 1

    correction_events = added + removed
    denominator = accepted + correction_events
    correction_rate = correction_events / denominator if denominator else 0.0

    return AnnotationStats(
        scenes_with_manual_annotations=len(manual_by_scene),
        generated_tag_assignments_on_annotated_scenes=generated_total,
        manual_tag_assignments=manual_total,
        accepted_ai_tag_assignments=accepted,
        human_added_tag_assignments=added,
        human_removed_generated_tag_assignments=removed,
        correction_events=correction_events,
        correction_rate=correction_rate,
        scenes_with_human_added_tags=scenes_added,
        scenes_with_removed_generated_tags=scenes_removed,
    )


def _load_json(path: Path, fallback: Any) -> Any:
    if not path.exists():
        return fallback
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise AnnotationStatsError(f"Invalid JSON: {path}: {exc}") from exc


def load_annotation_stats(metadata_dir: str | Path) -> AnnotationStats:
    """Load metadata files from disk and compute correction stats."""

    root = Path(metadata_dir)
    generated = _load_json(root / "scene_tags.json", {})
    manual = _load_json(root / MANUAL_ANNOTATIONS_FILENAME, {})
    if not isinstance(generated, dict):
        raise AnnotationStatsError("scene_tags.json must contain a tag index object")
    if not isinstance(manual, dict):
        raise AnnotationStatsError(f"{MANUAL_ANNOTATIONS_FILENAME} must contain an object")
    return compute_annotation_stats(generated, manual)
