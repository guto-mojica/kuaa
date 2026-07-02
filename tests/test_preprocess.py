"""Tests for the Pre-processing surface: cut-list model + review/edit service.

The scene-detection rebuild (which needs PySceneDetect + a real video) is
stubbed via ``_detector_for_rebuild`` so these tests exercise the cut-list
math, ``scene_cuts.json`` persistence, downstream invalidation, and the
filmstrip view model without decoding video.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kuaa.scene_detector import CutSet, SceneCut, read_cutset, write_cutset

# ── Cut-list model (pure) ─────────────────────────────────────────────────────


def test_cutset_scene_boundaries_and_counts():
    cs = CutSet(
        fps=24.0,
        total_frames=960,
        duration_s=40.0,
        cuts=[SceneCut(240, 10.0, "auto"), SceneCut(600, 25.0, "manual")],
    )
    assert cs.num_scenes == 3
    assert cs.scene_boundaries() == [(0, 240), (240, 600), (600, 960)]
    assert cs.has_manual_edits is True


def test_cutset_sorts_cuts_and_roundtrips(tmp_path):
    cs = CutSet(
        fps=24.0,
        total_frames=960,
        duration_s=40.0,
        cuts=[SceneCut(600, 25.0, "auto"), SceneCut(240, 10.0, "auto")],
    )
    # Boundaries are computed from sorted cuts regardless of insertion order.
    assert cs.scene_boundaries() == [(0, 240), (240, 600), (600, 960)]

    path = tmp_path / "scene_cuts.json"
    write_cutset(cs, path)
    back = read_cutset(path)
    assert back is not None
    assert back.total_frames == 960
    assert [c.frame for c in back.sorted_cuts()] == [240, 600]
    assert back.scene_boundaries() == cs.scene_boundaries()


def test_read_cutset_missing_returns_none(tmp_path):
    assert read_cutset(tmp_path / "nope.json") is None


# ── Service fixtures ──────────────────────────────────────────────────────────


class _FakeDetector:
    """Stand-in for SceneDetector.rebuild: writes keyframes_metadata.json
    consistent with the cut set's scene boundaries, no video decode."""

    def rebuild(self, cutset, video_path, keyframes_dir, metadata_path):
        Path(keyframes_dir).mkdir(parents=True, exist_ok=True)
        rows = []
        for i, (start, end) in enumerate(cutset.scene_boundaries(), start=1):
            kf = Path(keyframes_dir) / f"scene_{i:04d}_kf_01.jpg"
            kf.touch()
            rows.append(
                {
                    "scene_id": i,
                    "keyframe_id": f"scene_{i:04d}_kf_01",
                    "filepath": str(kf),
                    "start_time_s": start / cutset.fps,
                    "end_time_s": end / cutset.fps,
                    "duration_s": (end - start) / cutset.fps,
                    "start_frame": start,
                    "end_frame": end,
                }
            )
        Path(metadata_path).write_text(json.dumps(rows))
        return [], Path(metadata_path)


def _seed_three_scenes(seed_metadata):
    """Seed a 3-scene film with frame ranges + a matching scene_cuts.json."""
    scenes = [
        {
            "scene_id": 1,
            "filepath": "frames/s1.jpg",
            "start_time_s": 0.0,
            "end_time_s": 10.0,
            "start_frame": 0,
            "end_frame": 240,
        },
        {
            "scene_id": 2,
            "filepath": "frames/s2.jpg",
            "start_time_s": 10.0,
            "end_time_s": 25.0,
            "start_frame": 240,
            "end_frame": 600,
        },
        {
            "scene_id": 3,
            "filepath": "frames/s3.jpg",
            "start_time_s": 25.0,
            "end_time_s": 40.0,
            "start_frame": 600,
            "end_frame": 960,
        },
    ]
    paths = seed_metadata(scenes=scenes)
    cfg = paths["cfg"]
    from kuaa.library import FilmContext

    ctx = FilmContext.for_film(cfg, "default")
    cutset = CutSet(
        fps=24.0,
        total_frames=960,
        duration_s=40.0,
        cuts=[SceneCut(240, 10.0, "auto"), SceneCut(600, 25.0, "auto")],
        params={"keyframes_per_scene": 1, "keyframe_height": 480},
    )
    write_cutset(cutset, ctx.metadata_dir / "scene_cuts.json")
    return cfg, ctx


# ── build_filmstrip ───────────────────────────────────────────────────────────


def test_build_filmstrip_reports_scenes_and_stats(seed_metadata):
    cfg, ctx = _seed_three_scenes(seed_metadata)
    from kuaa.preprocess import build_filmstrip

    fs = build_filmstrip(ctx)
    assert fs["has_scenes"] is True
    assert fs["stats"]["num_scenes"] == 3
    assert [s["index"] for s in fs["scenes"]] == [1, 2, 3]
    # Scene 1 has no opening cut; scenes 2 and 3 open on auto cuts.
    assert fs["scenes"][0]["cut_source"] == ""
    assert fs["scenes"][1]["cut_source"] == "auto"
    assert fs["stats"]["min_s"] == 10.0 and fs["stats"]["max_s"] == 15.0


def test_build_filmstrip_empty_when_no_metadata(tmp_config):
    # Register an empty film with no keyframes_metadata.
    from kuaa.library import FilmContext, register_film
    from kuaa.preprocess import build_filmstrip

    library_dir = Path(tmp_config.paths.library_dir)
    register_film(library_dir, slug="empty", title="Empty", year=None, raw_filename="empty.mp4")
    ctx = FilmContext.for_film(tmp_config, "empty")
    fs = build_filmstrip(ctx)
    assert fs["has_scenes"] is False
    assert fs["scenes"] == []


# ── split / merge ─────────────────────────────────────────────────────────────


def test_merge_removes_cut_and_clears_downstream(seed_metadata, monkeypatch):
    cfg, ctx = _seed_three_scenes(seed_metadata)
    import kuaa.preprocess.service as svc

    monkeypatch.setattr(svc, "_detector_for_rebuild", lambda cfg, cutset: _FakeDetector())
    video = ctx.raw_path / "default.mp4"

    fs = svc.merge_at(ctx, cfg=cfg, video_path=video, cut_frame=240)
    assert fs["stats"]["num_scenes"] == 2

    # scene_cuts.json now has a single cut (600).
    cutset = read_cutset(ctx.metadata_dir / "scene_cuts.json")
    assert [c.frame for c in cutset.cuts] == [600]
    # Downstream artifacts cleared.
    assert not (ctx.metadata_dir / "visual_analysis.json").exists()
    assert not (ctx.metadata_dir / "scene_descriptions.json").exists()


def test_split_adds_manual_cut(seed_metadata, monkeypatch):
    cfg, ctx = _seed_three_scenes(seed_metadata)
    import kuaa.preprocess.service as svc

    monkeypatch.setattr(svc, "_detector_for_rebuild", lambda cfg, cutset: _FakeDetector())
    video = ctx.raw_path / "default.mp4"

    fs = svc.split_scene(ctx, cfg=cfg, video_path=video, at_frame=120)
    assert fs["stats"]["num_scenes"] == 4
    assert fs["has_manual_edits"] is True
    cutset = read_cutset(ctx.metadata_dir / "scene_cuts.json")
    assert 120 in [c.frame for c in cutset.cuts]
    assert any(c.frame == 120 and c.source == "manual" for c in cutset.cuts)


def test_split_at_existing_or_out_of_range_rejected(seed_metadata, monkeypatch):
    cfg, ctx = _seed_three_scenes(seed_metadata)
    import kuaa.preprocess.service as svc
    from kuaa.preprocess import CutEditError

    monkeypatch.setattr(svc, "_detector_for_rebuild", lambda cfg, cutset: _FakeDetector())
    video = ctx.raw_path / "default.mp4"

    with pytest.raises(CutEditError):
        svc.split_scene(ctx, cfg=cfg, video_path=video, at_frame=240)  # existing cut
    with pytest.raises(CutEditError):
        svc.split_scene(ctx, cfg=cfg, video_path=video, at_frame=0)  # bookend
    with pytest.raises(CutEditError):
        svc.split_scene(ctx, cfg=cfg, video_path=video, at_frame=5000)  # past end


def test_merge_unknown_cut_rejected(seed_metadata, monkeypatch):
    cfg, ctx = _seed_three_scenes(seed_metadata)
    import kuaa.preprocess.service as svc
    from kuaa.preprocess import CutEditError

    monkeypatch.setattr(svc, "_detector_for_rebuild", lambda cfg, cutset: _FakeDetector())
    video = ctx.raw_path / "default.mp4"
    with pytest.raises(CutEditError):
        svc.merge_at(ctx, cfg=cfg, video_path=video, cut_frame=999)


# ── Routes ────────────────────────────────────────────────────────────────────


def test_tab_pre_processing_renders(client, seed_metadata):
    seed_metadata()
    r = client.get("/tab/pre-processing?film=default")
    assert r.status_code == 200
    assert "pp-filmstrip" in r.text


def test_full_page_pre_processing_has_tab(client, seed_metadata):
    seed_metadata()
    r = client.get("/pre-processing")
    assert r.status_code == 200
    # Topbar carries the new tab.
    assert "/pre-processing" in r.text


def test_detect_unknown_film_rejected(client):
    r = client.post("/api/preprocess/detect", data={"film": "ghost"})
    assert r.status_code == 400


def test_cut_merge_unknown_film_404(client):
    r = client.post("/api/preprocess/cut/merge", data={"slug": "ghost", "cut_frame": 10})
    assert r.status_code == 404


def test_review_player_renders_when_source_present(client, seed_metadata):
    # seed_metadata touches library/default/raw/default.mp4, so the source
    # exists → the review player + split-at-playhead control render.
    seed_metadata()
    r = client.get("/tab/pre-processing?film=default")
    assert r.status_code == 200
    assert "pp-player__video" in r.text
    assert 'src="/api/preprocess/video/default"' in r.text
    assert "pp-split-here" in r.text


def test_video_route_serves_source_with_range(client, seed_metadata):
    paths = seed_metadata()
    raw = Path(paths["cfg"].paths.library_dir) / "default" / "raw" / "default.mp4"
    raw.write_bytes(b"\x00" * 4096)  # non-empty so a range is satisfiable
    r = client.get("/api/preprocess/video/default", headers={"Range": "bytes=0-1023"})
    assert r.status_code == 206  # partial content → seeking works
    assert r.headers["content-type"] == "video/mp4"
    assert r.headers.get("accept-ranges") == "bytes"


def test_video_route_unknown_film_404(client):
    r = client.get("/api/preprocess/video/ghost")
    assert r.status_code == 404


def test_job_scene_detection_only_classification():
    from api.jobs import JobState

    # A scene-detection-only run is owned by Pre-processing…
    sd = JobState(id="a", video_path="x.mp4", enabled_steps=frozenset({"scene_detection"}))
    assert sd.is_scene_detection_only is True
    # …a downstream Processing run is not, even though its steps list carries
    # every pipeline step name for the stepper UI.
    proc = JobState(id="b", video_path="x.mp4", enabled_steps=frozenset({"visual_analysis"}))
    assert proc.is_scene_detection_only is False
