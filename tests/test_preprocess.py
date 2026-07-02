"""Tests for the Pre-processing surface: cut-list model + review/edit service.

The scene-detection rebuild (which needs PySceneDetect + a real video) is
stubbed via ``_detector_for_rebuild`` so these tests exercise the cut-list
math, staged-edit persistence, the partial-rebuild rename/extract split, and
the filmstrip view model without decoding video.
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
    """Stand-in for SceneDetector: no video decode.

    Writes deterministic sentinel content so tests can distinguish a scene
    that was renamed (its original content survives) from one that was
    freshly "extracted" (this fake's sentinel appears instead). Tracks every
    call so tests can assert *which* boundaries actually needed a decode.
    """

    def __init__(self):
        self.calls: list[list[tuple[int, int]]] = []

    def extract_keyframes_for_boundaries(self, boundaries, scene_ids, fps, video_path, output_dir):
        self.calls.append(list(boundaries))
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out = {}
        for scene_id in scene_ids:
            p = output_dir / f"scene_{scene_id:04d}_kf_01.jpg"
            p.write_text(f"decoded-{scene_id}")
            out[scene_id] = [p]
        return out


def _seed_three_scenes(seed_metadata):
    """Seed a 3-scene film with real on-disk keyframe files + a matching
    scene_cuts.json. Real files (not path strings) so apply_pending's
    rename/extract swap has real inodes to move."""
    from kuaa.library import FilmContext

    paths = seed_metadata()
    cfg = paths["cfg"]
    ctx = FilmContext.for_film(cfg, "default")

    kf_dir = ctx.frames_dir / "scenes" / "keyframes_content"
    kf_dir.mkdir(parents=True, exist_ok=True)

    def _touch(scene_id: int) -> str:
        p = kf_dir / f"scene_{scene_id:04d}_kf_01.jpg"
        p.write_text(f"orig-{scene_id}")
        return str(p)

    scenes = [
        {
            "scene_id": 1,
            "filepath": _touch(1),
            "start_time_s": 0.0,
            "end_time_s": 10.0,
            "start_frame": 0,
            "end_frame": 240,
        },
        {
            "scene_id": 2,
            "filepath": _touch(2),
            "start_time_s": 10.0,
            "end_time_s": 25.0,
            "start_frame": 240,
            "end_frame": 600,
        },
        {
            "scene_id": 3,
            "filepath": _touch(3),
            "start_time_s": 25.0,
            "end_time_s": 40.0,
            "start_frame": 600,
            "end_frame": 960,
        },
    ]
    (ctx.metadata_dir / "keyframes_metadata.json").write_text(json.dumps(scenes))

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
    # No staged edits: nothing pending, no overlays.
    assert fs["pending_count"] == 0
    assert all(not s["pending_removal"] and not s["pending_splits"] for s in fs["scenes"])


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
    assert fs["pending_count"] == 0


# ── Staged edits (stage_split / stage_merge / toggle / discard) ────────────────


def test_stage_split_and_merge_do_not_rebuild(seed_metadata):
    cfg, ctx = _seed_three_scenes(seed_metadata)
    from kuaa.preprocess import build_filmstrip, has_pending, stage_merge, stage_split

    metadata_path = ctx.metadata_dir / "keyframes_metadata.json"
    before = metadata_path.read_text()

    assert has_pending(ctx) is False
    fs = stage_split(ctx, at_frame=120)
    assert has_pending(ctx) is True
    # No rebuild: keyframes_metadata.json is untouched, downstream is intact.
    assert metadata_path.read_text() == before
    assert (ctx.metadata_dir / "visual_analysis.json").exists()
    # Real tiles are unchanged; the live view reflects the staged split only
    # as an overlay + live stats, not a new tile.
    assert [s["index"] for s in fs["scenes"]] == [1, 2, 3]
    assert fs["stats"]["num_scenes"] == 4  # effective (staged) count
    assert fs["pending_count"] == 1
    assert fs["pending_ops"][0]["type"] == "split"

    fs = stage_merge(ctx, cut_frame=600)
    assert fs["pending_count"] == 2
    assert build_filmstrip(ctx)["stats"]["num_scenes"] == 3  # split then merge cancels out


def test_stage_split_rejects_bad_frame(seed_metadata):
    cfg, ctx = _seed_three_scenes(seed_metadata)
    from kuaa.preprocess import CutEditError, stage_split

    with pytest.raises(CutEditError):
        stage_split(ctx, at_frame=240)  # existing cut
    with pytest.raises(CutEditError):
        stage_split(ctx, at_frame=0)  # bookend
    with pytest.raises(CutEditError):
        stage_split(ctx, at_frame=5000)  # past end


def test_stage_merge_rejects_unknown_cut(seed_metadata):
    cfg, ctx = _seed_three_scenes(seed_metadata)
    from kuaa.preprocess import CutEditError, stage_merge

    with pytest.raises(CutEditError):
        stage_merge(ctx, cut_frame=999)


def test_toggle_pending_flips_one_op(seed_metadata):
    cfg, ctx = _seed_three_scenes(seed_metadata)
    from kuaa.preprocess import stage_merge, stage_split, toggle_pending

    stage_split(ctx, at_frame=120)
    fs = stage_merge(ctx, cut_frame=600)
    assert fs["stats"]["num_scenes"] == 3  # 3 + split(4) - merge(3)

    # Uncheck the merge: only the split's effect remains live.
    fs = toggle_pending(ctx, index=1)
    assert fs["pending_ops"][1]["checked"] is False
    assert fs["pending_ops"][0]["checked"] is True
    assert fs["stats"]["num_scenes"] == 4

    # Toggle it back on.
    fs = toggle_pending(ctx, index=1)
    assert fs["pending_ops"][1]["checked"] is True
    assert fs["stats"]["num_scenes"] == 3


def test_discard_pending_reverts_view(seed_metadata):
    cfg, ctx = _seed_three_scenes(seed_metadata)
    from kuaa.preprocess import build_filmstrip, discard_pending, has_pending, stage_split

    before = build_filmstrip(ctx)
    stage_split(ctx, at_frame=120)
    assert has_pending(ctx) is True

    fs = discard_pending(ctx)
    assert has_pending(ctx) is False
    assert fs["stats"] == before["stats"]
    assert fs["pending_count"] == 0


# ── replay() best-effort semantics (pure, no ctx needed) ───────────────────────


def test_replay_best_effort_semantics():
    from kuaa.preprocess.service import PendingOp, replay

    base = CutSet(
        fps=24.0,
        total_frames=960,
        duration_s=40.0,
        cuts=[SceneCut(240, 10.0, "auto"), SceneCut(600, 25.0, "auto")],
    )
    split = PendingOp(type="split", frame=120, time_s=5.0)
    cancel = PendingOp(type="merge", frame=120, time_s=5.0)

    # Both checked: split then its own cancelling merge — net no-op.
    result = replay(base, [split, cancel])
    assert [c.frame for c in result.sorted_cuts()] == [240, 600]

    # Only the merge checked (its originating split was unchecked/dropped):
    # best-effort skips it instead of raising, since frame 120 isn't a cut.
    result = replay(base, [cancel])
    assert [c.frame for c in result.sorted_cuts()] == [240, 600]

    # Only the split checked: applies normally.
    result = replay(base, [split])
    assert [c.frame for c in result.sorted_cuts()] == [120, 240, 600]


# ── apply_pending (partial rebuild) ─────────────────────────────────────────────


def test_apply_pending_errors_when_nothing_checked(seed_metadata, monkeypatch):
    cfg, ctx = _seed_three_scenes(seed_metadata)
    import kuaa.preprocess.service as svc
    from kuaa.preprocess import CutEditError, apply_pending, stage_split, toggle_pending

    stage_split(ctx, at_frame=120)
    toggle_pending(ctx, index=0)  # unchecked -> nothing to apply

    monkeypatch.setattr(svc, "_detector_for_rebuild", lambda cfg, cutset: _FakeDetector())
    with pytest.raises(CutEditError):
        apply_pending(ctx, cfg=cfg, video_path=ctx.raw_path / "default.mp4")


def test_apply_pending_renames_unchanged_reextracts_touched(seed_metadata, monkeypatch):
    cfg, ctx = _seed_three_scenes(seed_metadata)
    import kuaa.preprocess.service as svc
    from kuaa.preprocess import apply_pending, has_pending, stage_merge

    fake = _FakeDetector()
    monkeypatch.setattr(svc, "_detector_for_rebuild", lambda cfg, cutset: fake)
    video = ctx.raw_path / "default.mp4"

    # Merge scenes 2+3 (cut at 600): scene 1 is untouched (same boundary,
    # same final position) and should be renamed only, never re-extracted.
    stage_merge(ctx, cut_frame=600)
    fs = apply_pending(ctx, cfg=cfg, video_path=video)

    assert has_pending(ctx) is False
    assert fs["stats"]["num_scenes"] == 2
    # The fake detector only decoded the touched (merged) boundary.
    assert fake.calls == [[(240, 960)]]

    rows = json.loads((ctx.metadata_dir / "keyframes_metadata.json").read_text())
    by_scene = {r["scene_id"]: r for r in rows}
    assert Path(by_scene[1]["filepath"]).read_text() == "orig-1"  # renamed, not decoded
    assert Path(by_scene[2]["filepath"]).read_text() == "decoded-2"  # freshly extracted

    cutset = read_cutset(ctx.metadata_dir / "scene_cuts.json")
    assert [c.frame for c in cutset.cuts] == [240]
    assert not (ctx.metadata_dir / "visual_analysis.json").exists()
    assert not (ctx.metadata_dir / "scene_descriptions.json").exists()


def test_apply_pending_migrates_scene_id_overrides(seed_metadata, monkeypatch):
    cfg, ctx = _seed_three_scenes(seed_metadata)
    import kuaa.preprocess.service as svc
    from kuaa.preprocess import apply_pending, stage_split

    # Scenes 2 and 3 keep their boundaries but shift to positions 3 and 4
    # once scene 1 is split in two; scene 1's own override has no surviving
    # boundary and should be dropped.
    (ctx.metadata_dir / "manual_annotations.json").write_text(
        json.dumps({"1": ["stale"], "2": ["kept-two"], "3": ["kept-three"]})
    )
    (ctx.metadata_dir / "tag_overrides.json").write_text(
        json.dumps({"1": {"suppressed": ["stale"]}, "2": {"suppressed": ["kept"]}})
    )

    monkeypatch.setattr(svc, "_detector_for_rebuild", lambda cfg, cutset: _FakeDetector())
    stage_split(ctx, at_frame=120)
    apply_pending(ctx, cfg=cfg, video_path=ctx.raw_path / "default.mp4")

    annotations = json.loads((ctx.metadata_dir / "manual_annotations.json").read_text())
    assert annotations == {"3": ["kept-two"], "4": ["kept-three"]}
    overrides = json.loads((ctx.metadata_dir / "tag_overrides.json").read_text())
    assert overrides == {"3": {"suppressed": ["kept"]}}


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


def test_cut_apply_unknown_film_404(client):
    r = client.post("/api/preprocess/cut/apply", data={"slug": "ghost"})
    assert r.status_code == 404


def test_cut_discard_unknown_film_404(client):
    r = client.post("/api/preprocess/cut/discard", data={"slug": "ghost"})
    assert r.status_code == 404


def test_cut_pending_toggle_unknown_film_404(client):
    r = client.post("/api/preprocess/cut/pending/0/toggle", data={"slug": "ghost"})
    assert r.status_code == 404


def test_cut_routes_stage_and_apply_end_to_end(client, seed_metadata, monkeypatch):
    # Route-level smoke test: stage a split via HTTP, see it listed as
    # pending, then apply it via HTTP. ``client`` and ``seed_metadata`` share
    # the same ``tmp_config`` the routes read via ``get_config()``, so a ctx
    # built straight from ``_seed_three_scenes`` is exactly what the routes
    # will resolve for slug "default".
    cfg, ctx = _seed_three_scenes(seed_metadata)

    import kuaa.preprocess.service as svc

    monkeypatch.setattr(svc, "_detector_for_rebuild", lambda cfg, cutset: _FakeDetector())

    r = client.post("/api/preprocess/cut/split", data={"slug": "default", "at_frame": 120})
    assert r.status_code == 200
    assert "pp-pending" in r.text

    r = client.post("/api/preprocess/cut/apply", data={"slug": "default"})
    assert r.status_code == 200
    assert "pp-pending" not in r.text  # nothing pending after apply


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
