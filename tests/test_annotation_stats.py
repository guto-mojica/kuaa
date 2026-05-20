from __future__ import annotations

import json

import pytest

from cinemateca.eval.annotations import (
    AnnotationStatsError,
    compute_annotation_stats,
    load_annotation_stats,
    normalize_tag,
    tag_index_to_scene_tags,
)


def test_normalize_tag_matches_manual_save_shape():
    assert normalize_tag("  Train Car  ") == "train-car"


def test_tag_index_to_scene_tags_normalizes_mixed_scene_ids():
    by_scene = tag_index_to_scene_tags({"Exterior": [1, "2", 2.0]})

    assert by_scene == {"1": {"exterior"}, "2": {"exterior"}}


def test_compute_annotation_stats_reports_overlap_additions_and_removals():
    generated = {
        "exterior": [1, 2],
        "train car": [1],
        "people": [2],
        "remove-me": [3],
    }
    manual = {
        "1": ["exterior", "human note"],
        "2": ["people", "exterior"],
        "3": ["only manual"],
    }

    stats = compute_annotation_stats(generated, manual)

    assert stats.scenes_with_manual_annotations == 3
    assert stats.generated_tag_assignments_on_annotated_scenes == 5
    assert stats.manual_tag_assignments == 5
    assert stats.accepted_ai_tag_assignments == 3
    assert stats.human_added_tag_assignments == 2
    assert stats.human_removed_generated_tag_assignments == 2
    assert stats.correction_events == 4
    assert stats.correction_rate == pytest.approx(4 / 7)
    assert stats.scenes_with_human_added_tags == 2
    assert stats.scenes_with_removed_generated_tags == 2


def test_load_annotation_stats_missing_files_returns_zero_stats(tmp_path):
    stats = load_annotation_stats(tmp_path)

    assert stats.to_dict()["scenes_with_manual_annotations"] == 0
    assert stats.correction_rate == 0.0


def test_load_annotation_stats_rejects_invalid_json(tmp_path):
    (tmp_path / "scene_tags.json").write_text("{", encoding="utf-8")
    (tmp_path / "manual_annotations.json").write_text(json.dumps({}), encoding="utf-8")

    with pytest.raises(AnnotationStatsError, match="Invalid JSON"):
        load_annotation_stats(tmp_path)
