"""Per-scene motion statistics.

Validated against synthetic video with known ground truth: a locked-off
shot, a horizontal pan, a vertical tilt, and a subject moving across a
static frame. Synthetic rather than real footage because the point is
whether the classifier recovers a motion we constructed — with archive
material there is nothing to check the answer against.
"""

from __future__ import annotations

import numpy as np
import pytest

from kuaa.motion import SceneMotion, analyze_video, load_motion, motion_tag_index, save_motion

cv2 = pytest.importorskip("cv2")

FPS = 24.0
W, H = 640, 480
SCENE_FRAMES = 48


def _texture(seed: int = 0) -> np.ndarray:
    """A deterministic high-frequency field — optical flow needs texture."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(H * 3, W * 3), dtype=np.uint8)
    return cv2.cvtColor(cv2.GaussianBlur(base, (5, 5), 0), cv2.COLOR_GRAY2BGR)


def _write_video(path, frames: list[np.ndarray]) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    assert writer.isOpened(), "could not open a VideoWriter for the fixture"
    for frame in frames:
        writer.write(frame)
    writer.release()


def _crop(canvas: np.ndarray, x: int, y: int) -> np.ndarray:
    return canvas[y : y + H, x : x + W].copy()


def _static_frames() -> list[np.ndarray]:
    canvas = _texture()
    return [_crop(canvas, W, H) for _ in range(SCENE_FRAMES)]


def _pan_frames(dx: int = 6) -> list[np.ndarray]:
    canvas = _texture()
    return [_crop(canvas, W + i * dx, H) for i in range(SCENE_FRAMES)]


def _tilt_frames(dy: int = 6) -> list[np.ndarray]:
    canvas = _texture()
    return [_crop(canvas, W, H + i * dy) for i in range(SCENE_FRAMES)]


def _subject_frames() -> list[np.ndarray]:
    """Static camera, one bright block travelling across the frame."""
    canvas = _texture()
    out = []
    for i in range(SCENE_FRAMES):
        frame = _crop(canvas, W, H)
        x = 20 + i * 9
        frame[180:300, x : x + 120] = 255
        out.append(frame)
    return out


@pytest.fixture
def four_scene_video(tmp_path):
    """One file, four scenes: static, pan, tilt, moving subject."""
    frames = _static_frames() + _pan_frames() + _tilt_frames() + _subject_frames()
    path = tmp_path / "fixture.avi"
    _write_video(path, frames)
    boundaries = [
        (0, SCENE_FRAMES),
        (SCENE_FRAMES, SCENE_FRAMES * 2),
        (SCENE_FRAMES * 2, SCENE_FRAMES * 3),
        (SCENE_FRAMES * 3, SCENE_FRAMES * 4),
    ]
    return path, boundaries


def _records(four_scene_video) -> list[SceneMotion]:
    path, boundaries = four_scene_video
    return analyze_video(path, boundaries, FPS, sample_fps=8.0)


# ── the classifier recovers what was constructed ─────────────────────────────


def test_one_record_per_scene(four_scene_video) -> None:
    records = _records(four_scene_video)
    assert [r.scene_id for r in records] == [1, 2, 3, 4]


def test_locked_off_shot_reads_as_static(four_scene_video) -> None:
    static = _records(four_scene_video)[0]
    assert static.camera_motion == "static"
    assert static.subject_motion == "still"
    assert static.motion_magnitude < 0.1


def test_horizontal_pan_is_distinguished_from_a_static_shot(four_scene_video) -> None:
    records = _records(four_scene_video)
    static, pan = records[0], records[1]
    assert pan.camera_motion == "pan"
    assert pan.motion_magnitude > static.motion_magnitude


def test_vertical_tilt_is_distinguished_from_a_pan(four_scene_video) -> None:
    tilt = _records(four_scene_video)[2]
    assert tilt.camera_motion == "tilt"


def test_subject_motion_is_separated_from_camera_motion(four_scene_video) -> None:
    """The crossing with object detection depends on this separation.

    A block moving across an otherwise locked-off frame must register as
    subject motion, not as a camera move — that distinction is what makes
    `horse` + moving mean something different from `horse` + still.
    """
    subject = _records(four_scene_video)[3]
    assert subject.subject_motion == "moving"
    assert subject.camera_share < 0.6


def test_camera_moves_are_attributed_to_the_camera(four_scene_video) -> None:
    pan = _records(four_scene_video)[1]
    assert pan.camera_share >= 0.6


def test_shot_length_comes_from_the_boundaries(four_scene_video) -> None:
    for record in _records(four_scene_video):
        assert record.shot_length_s == pytest.approx(SCENE_FRAMES / FPS, abs=0.01)


# ── degradation ──────────────────────────────────────────────────────────────


def test_unreadable_video_raises_oserror(tmp_path) -> None:
    with pytest.raises(OSError):
        analyze_video(tmp_path / "missing.avi", [(0, 10)], FPS)


def test_scene_past_the_end_degrades_instead_of_raising(four_scene_video) -> None:
    """One undecodable region must not fail the whole film."""
    path, _ = four_scene_video
    records = analyze_video(path, [(0, 48), (10_000, 10_100)], FPS, sample_fps=8.0)
    assert records[0].sampled_pairs > 0
    assert records[1].sampled_pairs == 0
    assert records[1].camera_motion == "static"


def test_long_takes_are_capped(tmp_path) -> None:
    """A 186s tableau should not cost 700 flow fields to call static."""
    path = tmp_path / "long.avi"
    _write_video(path, _static_frames())
    records = analyze_video(path, [(0, SCENE_FRAMES)], FPS, sample_fps=24.0, max_pairs_per_scene=4)
    assert records[0].sampled_pairs <= 4


# ── tags ─────────────────────────────────────────────────────────────────────


def test_tags_are_portuguese_and_describe_the_motion(four_scene_video) -> None:
    records = _records(four_scene_video)
    assert "plano-estatico" in records[0].tags()
    assert "camera-panoramica" in records[1].tags()
    assert "camera-vertical" in records[2].tags()
    assert "movimento-de-cena" in records[3].tags()


def test_unmeasured_scenes_produce_no_tags() -> None:
    """Better to say nothing than to label a scene we could not decode."""
    record = SceneMotion(
        scene_id=1,
        shot_length_s=5.0,
        motion_magnitude=0.0,
        camera_motion="static",
        subject_motion="still",
        camera_share=0.0,
        sampled_pairs=0,
    )
    assert record.tags() == []


def test_long_shots_are_tagged() -> None:
    record = SceneMotion(
        scene_id=1,
        shot_length_s=45.0,
        motion_magnitude=0.1,
        camera_motion="static",
        subject_motion="still",
        camera_share=0.9,
        sampled_pairs=10,
    )
    assert "plano-longo" in record.tags()


def test_motion_tag_index_has_the_scene_tags_shape(four_scene_video) -> None:
    index = motion_tag_index(_records(four_scene_video))
    assert index["plano-estatico"] == [1]
    assert all(isinstance(sids, list) for sids in index.values())


# ── the artefact ─────────────────────────────────────────────────────────────


def test_round_trips_through_disk(tmp_path, four_scene_video) -> None:
    # Floats are rounded to 4dp on write so the artefact stays readable;
    # compare with that tolerance rather than demanding bit-exactness.
    records = _records(four_scene_video)
    save_motion(tmp_path, records)
    loaded = load_motion(tmp_path)
    assert len(loaded) == len(records)
    for got, want in zip(loaded, records, strict=True):
        assert got.scene_id == want.scene_id
        assert got.camera_motion == want.camera_motion
        assert got.subject_motion == want.subject_motion
        assert got.sampled_pairs == want.sampled_pairs
        assert got.motion_magnitude == pytest.approx(want.motion_magnitude, abs=1e-4)
        assert got.camera_share == pytest.approx(want.camera_share, abs=1e-4)
        assert got.shot_length_s == pytest.approx(want.shot_length_s, abs=1e-4)


def test_absent_artefact_loads_as_empty(tmp_path) -> None:
    assert load_motion(tmp_path) == []


def test_motion_separates_scenes_that_are_otherwise_identical(tmp_path) -> None:
    """The payoff: motion crossed with object detection.

    Two scenes carry the same description and the same object tag. Nothing
    in the still-image pipeline can tell them apart. Adding the motion
    surface makes ``cavalo camera panoramica`` prefer the one that pans.
    """
    import json

    from kuaa.search.bm25 import bm25_index_for_dir

    metadata_dir = tmp_path / "film" / "metadata"
    metadata_dir.mkdir(parents=True)

    # Only scene 2 pans; a realistic distribution, not 50/50 — BM25 IDF
    # goes to zero for a term carried by exactly half the corpus.
    frames: list[np.ndarray] = []
    boundaries: list[tuple[int, int]] = []
    for scene in range(1, 9):
        start = len(frames)
        frames += _pan_frames() if scene == 2 else _static_frames()
        boundaries.append((start, len(frames)))
    video = tmp_path / "film.avi"
    _write_video(video, frames)
    save_motion(metadata_dir, analyze_video(video, boundaries, FPS, sample_fps=8.0))

    descriptions = [
        {"scene_id": 1, "description": "a man on a horse in a field"},
        {"scene_id": 2, "description": "a man on a horse in a field"},
    ]
    descriptions += [
        {"scene_id": s, "description": f"an interior room with a table {s}"} for s in range(3, 9)
    ]
    (metadata_dir / "scene_descriptions.json").write_text(json.dumps(descriptions))
    (metadata_dir / "scene_tags.json").write_text(json.dumps({"horse": [1, 2]}))

    index = bm25_index_for_dir(
        metadata_dir=metadata_dir,
        stopwords_lang=None,
        k1=1.5,
        b=0.75,
        tokenizer_name="multilingual",
        bilingual=True,
    )
    panning = [sid for sid, _ in index.query("cavalo camera panoramica", 3)]
    assert panning[0] == 2, "the panning scene should win a panning query"
    plain = [sid for sid, _ in index.query("cavalo", 3)]
    assert set(plain[:2]) == {1, 2}, "both horse scenes still match the plain query"


def test_malformed_artefact_loads_as_empty(tmp_path) -> None:
    (tmp_path / "scene_motion.json").write_text("{not json")
    assert load_motion(tmp_path) == []
