"""The visual_analysis → scene_id join, across both keyframe naming conventions.

``visual_analysis.json`` rows identify their frame only by a ``frame_path``
basename and carry no ``scene_id``. Resolving one therefore depends on the
``keyframes_metadata.json`` manifest. Before that join existed, the Scenes tab
showed no detector output for any film and the object leg of the search scorer
contributed nothing for films using the ``scene_0001_kf_01.jpg`` convention.
"""

from __future__ import annotations

import json
from pathlib import Path

from kuaa.library import frame_to_scene_index, load_metadata
from kuaa.search._aggregate.coco_aliases import with_coco_aliases
from kuaa.search._aggregate.scorers import MetadataScorer, _scene_id_from_visual_record

# The two conventions live side by side in the real library.
_MODERN = "scene_0001_kf_01.jpg"
_LEGACY = "a-film-Scene-001-01.jpg"


def _kf_meta(*names: str) -> list[dict]:
    return [
        {"scene_id": i + 1, "filepath": f"/lib/a/frames/keyframes_content/{n}"}
        for i, n in enumerate(names)
    ]


def test_frame_to_scene_index_keys_on_basename() -> None:
    index = frame_to_scene_index(_kf_meta(_MODERN, _LEGACY))
    assert index == {_MODERN: 1, _LEGACY: 2}


def test_frame_to_scene_index_skips_incomplete_rows() -> None:
    rows: list = [
        {"scene_id": 1, "filepath": "/lib/a/x.jpg"},
        {"scene_id": 2},  # no filepath
        {"filepath": "/lib/a/y.jpg"},  # no scene_id
        "not-a-dict",
    ]
    assert frame_to_scene_index(rows) == {"x.jpg": 1}


def test_scene_id_resolves_both_naming_conventions() -> None:
    """The manifest carries both; the legacy regex alone only ever saw one."""
    index = frame_to_scene_index(_kf_meta(_MODERN, _LEGACY))
    assert _scene_id_from_visual_record({"frame_path": _MODERN}, index) == 1
    assert _scene_id_from_visual_record({"frame_path": _LEGACY}, index) == 2

    # Without the manifest, only the hyphenated legacy name resolves. This is
    # the bug the join fixes, pinned so a regression is unambiguous.
    assert _scene_id_from_visual_record({"frame_path": _MODERN}) is None
    assert _scene_id_from_visual_record({"frame_path": _LEGACY}) == 1


def test_scene_id_prefers_explicit_field_then_manifest_then_regex() -> None:
    index = {_MODERN: 7}
    assert _scene_id_from_visual_record({"scene_id": 3, "frame_path": _MODERN}, index) == 3
    assert _scene_id_from_visual_record({"frame_path": _MODERN}, index) == 7
    # Manifest miss falls through to the regex rather than dropping the row.
    assert _scene_id_from_visual_record({"frame_path": _LEGACY}, index) == 1


def test_load_metadata_surfaces_visual_rows_without_scene_id(tmp_path: Path) -> None:
    """Regression: ``vis_by_scene`` was empty for every film in the library."""
    md = tmp_path / "metadata"
    md.mkdir()
    (md / "keyframes_metadata.json").write_text(json.dumps(_kf_meta(_MODERN, _LEGACY)))
    (md / "visual_analysis.json").write_text(
        json.dumps(
            [
                {"frame_path": _MODERN, "face_detection": {"num_faces": 2}},
                {"frame_path": _LEGACY, "face_detection": {"num_faces": 0}},
                {"frame_path": "orphan.jpg", "face_detection": {"num_faces": 9}},
            ]
        )
    )
    _kf, _desc, vis_by_scene, _tags = load_metadata(md)

    assert set(vis_by_scene) == {"1", "2"}
    assert vis_by_scene["1"]["face_detection"]["num_faces"] == 2
    # A frame absent from the manifest cannot be placed and is dropped.
    assert all(v["face_detection"]["num_faces"] != 9 for v in vis_by_scene.values())


def test_load_metadata_keeps_first_row_of_a_multi_keyframe_scene(tmp_path: Path) -> None:
    md = tmp_path / "metadata"
    md.mkdir()
    (md / "keyframes_metadata.json").write_text(
        json.dumps(
            [
                {"scene_id": 1, "filepath": "/lib/a/scene_0001_kf_01.jpg"},
                {"scene_id": 1, "filepath": "/lib/a/scene_0001_kf_02.jpg"},
            ]
        )
    )
    (md / "visual_analysis.json").write_text(
        json.dumps(
            [
                {"frame_path": "scene_0001_kf_01.jpg", "environment": {"location": "interior"}},
                {"frame_path": "scene_0001_kf_02.jpg", "environment": {"location": "exterior"}},
            ]
        )
    )
    _kf, _desc, vis_by_scene, _tags = load_metadata(md)
    assert list(vis_by_scene) == ["1"]
    assert vis_by_scene["1"]["environment"]["location"] == "interior"


# ── Detector object scoring ──────────────────────────────────────────────────


def _visual_rows() -> list[dict]:
    return [
        {
            "frame_path": _MODERN,
            "object_detection": {"objects": [{"class": "horse"}], "class_counts": {"horse": 1}},
        }
    ]


def test_metadata_scorer_scores_objects_only_with_the_manifest() -> None:
    """The object signal is dead without the join for modern-convention films."""
    index = frame_to_scene_index(_kf_meta(_MODERN))
    scorer = MetadataScorer()
    kwargs: dict = {"descriptions": [], "tag_index": {}, "visual_rows": _visual_rows()}

    assert scorer.score(query="horse", frame_to_scene=index, **kwargs) == {1: 20.0}
    assert scorer.score(query="horse", **kwargs) == {}


def test_metadata_scorer_matches_portuguese_query_against_english_class() -> None:
    """The detector emits COCO English; the interface language is pt-BR."""
    index = frame_to_scene_index(_kf_meta(_MODERN))
    scorer = MetadataScorer()
    kwargs: dict = {
        "descriptions": [],
        "tag_index": {},
        "visual_rows": _visual_rows(),
        "frame_to_scene": index,
    }
    assert scorer.score(query="cavalo", **kwargs) == scorer.score(query="horse", **kwargs)
    assert scorer.score(query="cavalo", **kwargs) == {1: 20.0}
    # An unaliased word must not become a match for something else.
    assert scorer.score(query="bicicleta", **kwargs) == {}


def test_coco_aliases_pass_english_through_and_expand_multiword() -> None:
    assert with_coco_aliases(["horse"]) == ["horse"]
    assert with_coco_aliases(["cavalo"]) == ["horse"]
    assert with_coco_aliases(["planta"]) == ["potted", "plant"]
    assert with_coco_aliases(["zebra", "gato"]) == ["zebra", "cat"]
    assert with_coco_aliases([]) == []
