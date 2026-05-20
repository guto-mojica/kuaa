"""Tests for ``SceneDetector.export_metadata`` after the 1:N density fix.

Prior to Phase 1 of the density plan, ``export_metadata`` emitted a
single JSON row per detected scene, pointing at the *middle* keyframe
and discarding the start/end keyframes that PySceneDetect had already
saved to disk. For tableau-style films (1903 *Great Train Robbery* —
7 scenes, one 5.5-min single shot) this collapsed the entire film to
7 CLIP vectors and made per-film search degenerate (``top_k=8 >=
n_vectors=7`` => every query returns the whole corpus).

The fix emits one metadata row per *saved keyframe* (N rows per scene),
all sharing ``scene_id`` and scene-level time/frame fields but with
distinct ``keyframe_id`` and ``filepath``. Downstream search dedupes
back to one result per scene using max(similarity).

These tests pin the 1:N row shape and the field-sharing contract.
"""
from __future__ import annotations

import json

import pytest


def _fake_timecode(seconds: float, fps: float = 24.0):
    """A stand-in for PySceneDetect's Timecode object.

    Only ``.get_seconds()`` and ``.get_frames()`` are called by
    ``export_metadata`` plus arithmetic (``end - start``) for duration.
    Using SimpleNamespace with the two methods is enough; duration is
    computed by the caller via ``(end - start).get_seconds()`` so we
    expose ``__sub__`` to return another fake.
    """
    class _T:
        def __init__(self, s: float):
            self._s = s
        def get_seconds(self) -> float:
            return self._s
        def get_frames(self) -> int:
            return int(self._s * fps)
        def __sub__(self, other):
            return _T(self._s - other._s)
    return _T(seconds)


@pytest.fixture()
def fake_scene_list():
    """Two scenes: 0-10s and 10-310s (a long 5-minute tableau shot)."""
    return [
        (_fake_timecode(0.0), _fake_timecode(10.0)),
        (_fake_timecode(10.0), _fake_timecode(310.0)),
    ]


@pytest.fixture()
def fake_keyframe_paths(tmp_path):
    """Six keyframe paths — 3 per scene — mimicking PySceneDetect's
    Scene-NNN-KK.jpg ordering."""
    paths = []
    for scene in (1, 2):
        for kf in (1, 2, 3):
            p = tmp_path / f"Scene-{scene:03d}-{kf:02d}.jpg"
            p.touch()
            paths.append(p)
    return paths


# ── 1:N row emission ──────────────────────────────────────────────────────────

class TestExportMetadataOneToMany:
    """``export_metadata`` emits N rows per scene (N = keyframes_per_scene),
    each row sharing scene-level fields but with a unique keyframe."""

    def test_six_keyframes_two_scenes_yields_six_rows(
        self, fake_scene_list, fake_keyframe_paths, tmp_path
    ):
        from cinemateca.scene_detector import SceneDetector

        det = SceneDetector()
        det.keyframes_per_scene = 3
        out = tmp_path / "keyframes_metadata.json"
        det.export_metadata(fake_scene_list, fake_keyframe_paths, out)

        data = json.loads(out.read_text())
        assert len(data) == 6, (
            f"expected 6 rows (2 scenes × 3 keyframes), got {len(data)}"
        )

    def test_rows_for_same_scene_share_scene_id_and_times(
        self, fake_scene_list, fake_keyframe_paths, tmp_path
    ):
        from cinemateca.scene_detector import SceneDetector

        det = SceneDetector()
        det.keyframes_per_scene = 3
        out = tmp_path / "keyframes_metadata.json"
        det.export_metadata(fake_scene_list, fake_keyframe_paths, out)
        data = json.loads(out.read_text())

        scene1 = [r for r in data if r["scene_id"] == 1]
        scene2 = [r for r in data if r["scene_id"] == 2]
        assert len(scene1) == 3 and len(scene2) == 3

        for rows, start, end in ((scene1, 0.0, 10.0), (scene2, 10.0, 310.0)):
            assert {r["start_time_s"] for r in rows} == {start}
            assert {r["end_time_s"] for r in rows} == {end}
            assert {r["duration_s"] for r in rows} == {end - start}

    def test_rows_for_same_scene_differ_in_keyframe_id_and_filepath(
        self, fake_scene_list, fake_keyframe_paths, tmp_path
    ):
        from cinemateca.scene_detector import SceneDetector

        det = SceneDetector()
        det.keyframes_per_scene = 3
        out = tmp_path / "keyframes_metadata.json"
        det.export_metadata(fake_scene_list, fake_keyframe_paths, out)
        data = json.loads(out.read_text())

        # keyframe_id must be globally unique
        kf_ids = [r["keyframe_id"] for r in data]
        assert len(kf_ids) == len(set(kf_ids)), (
            f"keyframe_ids must be unique: {kf_ids}"
        )

        # ...and follow the documented naming convention
        for r in data:
            assert r["keyframe_id"].startswith(
                f"scene_{r['scene_id']:04d}_kf_"
            ), r["keyframe_id"]

        # Each row's filepath must be one of the six on-disk JPGs.
        filepaths = {r["filepath"] for r in data}
        assert filepaths == {str(p) for p in fake_keyframe_paths}

    def test_kf_per_scene_one_is_backward_compatible(
        self, fake_scene_list, tmp_path
    ):
        """With keyframes_per_scene=1, the output is 1 row per scene —
        same as the legacy behavior (only keyframe_id naming changed)."""
        from cinemateca.scene_detector import SceneDetector

        paths = []
        for scene in (1, 2):
            p = tmp_path / f"Scene-{scene:03d}-01.jpg"
            p.touch()
            paths.append(p)

        det = SceneDetector()
        det.keyframes_per_scene = 1
        out = tmp_path / "keyframes_metadata.json"
        det.export_metadata(fake_scene_list, paths, out)
        data = json.loads(out.read_text())

        assert len(data) == 2
        assert data[0]["scene_id"] == 1
        assert data[1]["scene_id"] == 2
        assert all(r["keyframe_id"].endswith("_kf_01") for r in data)
