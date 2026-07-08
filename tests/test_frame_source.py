"""FrameSource Protocol + registry + directory_stills backend (Seam 1).

Hermetic: no video, no GPU, no heavy model deps. Exercises the input-side
seam — the ``video_scenedetect`` default resolves without importing
PySceneDetect (its import is deferred into ``produce``), and
``directory_stills`` runs end-to-end using Pillow only.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke


def _models_cfg(frame_source: str = "video_scenedetect", keyframe_height: int = 480):
    """Minimal cfg stub with just the sections the seam reads."""
    sn = types.SimpleNamespace
    return sn(
        models=sn(frame_source=frame_source),
        scene_detection=sn(keyframe_height=keyframe_height),
    )


# ─── Protocol ─────────────────────────────────────────────────────────────────


def test_frame_source_protocol_runtime_checkable():
    from kuaa.models.base import FrameSource

    assert getattr(FrameSource, "_is_runtime_protocol", False)


def test_frame_source_structural_isinstance():
    from kuaa.models.base import FrameSource

    class _Good:
        def produce(self, source, *, keyframes_dir, metadata_path, cuts_path=None):
            return {}

    class _Bad:
        pass

    assert isinstance(_Good(), FrameSource) is True
    assert isinstance(_Bad(), FrameSource) is False
    members = {m for m in vars(FrameSource) if not m.startswith("_")}
    assert "produce" in members


# ─── Registry ─────────────────────────────────────────────────────────────────


def test_registry_returns_video_default():
    from kuaa.models import registry
    from kuaa.models.base import FrameSource
    from kuaa.models.frame_source.video_scenedetect import VideoSceneDetectFrameSource

    src = registry.get_frame_source(_models_cfg("video_scenedetect"))
    assert isinstance(src, FrameSource)
    assert isinstance(src, VideoSceneDetectFrameSource)


def test_registry_returns_directory_stills():
    from kuaa.models import registry
    from kuaa.models.base import FrameSource
    from kuaa.models.frame_source.directory_stills import DirectoryStillsFrameSource

    src = registry.get_frame_source(_models_cfg("directory_stills"))
    assert isinstance(src, FrameSource)
    assert isinstance(src, DirectoryStillsFrameSource)


def test_registry_unknown_frame_source_raises():
    from kuaa.models import registry

    with pytest.raises(ValueError):
        registry.get_frame_source(_models_cfg("nope"))


# ─── directory_stills backend ─────────────────────────────────────────────────


def test_directory_stills_produces_video_compatible_manifest(tmp_path):
    from PIL import Image

    from kuaa.models.frame_source.directory_stills import DirectoryStillsFrameSource

    src_dir = tmp_path / "stills"
    src_dir.mkdir()
    Image.new("RGB", (800, 600), "red").save(src_dir / "b.png")
    Image.new("RGB", (400, 400), "blue").save(src_dir / "a.jpg")
    (src_dir / "notes.txt").write_text("not an image")  # must be ignored

    kf_dir = tmp_path / "frames" / "scenes" / "keyframes_content"
    meta_path = tmp_path / "keyframes_metadata.json"

    backend = DirectoryStillsFrameSource(_models_cfg("directory_stills", keyframe_height=480))
    manifest = backend.produce(src_dir, keyframes_dir=kf_dir, metadata_path=meta_path)

    # Manifest shape mirrors the video path's step output.
    assert manifest["metadata_path"] == meta_path
    assert manifest["keyframes_dir"] == kf_dir
    assert manifest["stats"]["num_scenes"] == 2
    assert len(manifest["keyframes"]) == 2

    rows = json.loads(meta_path.read_text(encoding="utf-8"))
    assert [r["scene_id"] for r in rows] == [1, 2]
    assert [r["keyframe_id"] for r in rows] == ["scene_0001_kf_01", "scene_0002_kf_01"]
    # Deterministic order: sorted by filename → a.jpg is scene 1, b.png scene 2.
    assert Path(rows[0]["source_path"]).name == "a.jpg"
    assert Path(rows[1]["source_path"]).name == "b.png"
    for r in rows:
        p = Path(r["filepath"])
        assert p.exists() and p.suffix == ".jpg"
        assert r["start_time_s"] == 0.0
        assert r["end_time_s"] == 0.0
        assert r["duration_s"] == 0.0

    # Canonical keyframe filenames land in the keyframes dir as 8-bit RGB JPEG.
    assert (kf_dir / "scene_0001_kf_01.jpg").exists()
    with Image.open(kf_dir / "scene_0002_kf_01.jpg") as im:
        assert im.mode == "RGB"
        assert im.height == 480  # downscaled to keyframe_height
        assert im.width == 640  # 800 * (480 / 600), aspect preserved


def test_directory_stills_no_downscale_when_height_zero(tmp_path):
    from PIL import Image

    from kuaa.models.frame_source.directory_stills import DirectoryStillsFrameSource

    src_dir = tmp_path / "stills"
    src_dir.mkdir()
    Image.new("RGB", (320, 200), "green").save(src_dir / "img.png")

    backend = DirectoryStillsFrameSource(_models_cfg("directory_stills", keyframe_height=0))
    backend.produce(
        src_dir,
        keyframes_dir=tmp_path / "kf",
        metadata_path=tmp_path / "m.json",
    )
    with Image.open(tmp_path / "kf" / "scene_0001_kf_01.jpg") as im:
        assert (im.width, im.height) == (320, 200)


def test_directory_stills_empty_dir_raises(tmp_path):
    from kuaa.models.frame_source.directory_stills import DirectoryStillsFrameSource

    empty = tmp_path / "empty"
    empty.mkdir()
    backend = DirectoryStillsFrameSource(_models_cfg("directory_stills"))
    with pytest.raises(FileNotFoundError):
        backend.produce(empty, keyframes_dir=tmp_path / "kf", metadata_path=tmp_path / "m.json")


def test_directory_stills_missing_source_raises(tmp_path):
    from kuaa.models.frame_source.directory_stills import DirectoryStillsFrameSource

    backend = DirectoryStillsFrameSource(_models_cfg("directory_stills"))
    with pytest.raises(FileNotFoundError):
        backend.produce(
            tmp_path / "does_not_exist",
            keyframes_dir=tmp_path / "kf",
            metadata_path=tmp_path / "m.json",
        )


def test_directory_stills_file_source_raises(tmp_path):
    from PIL import Image

    from kuaa.models.frame_source.directory_stills import DirectoryStillsFrameSource

    f = tmp_path / "single.jpg"
    Image.new("RGB", (10, 10), "red").save(f)
    backend = DirectoryStillsFrameSource(_models_cfg("directory_stills"))
    with pytest.raises(NotADirectoryError):
        backend.produce(f, keyframes_dir=tmp_path / "kf", metadata_path=tmp_path / "m.json")


# ─── Config selector ──────────────────────────────────────────────────────────


def test_config_default_frame_source_is_video():
    from kuaa.config import load_config

    cfg = load_config(project_root=".", ensure_dirs=False)
    assert cfg.models.frame_source == "video_scenedetect"


def test_config_frame_source_literal_enforced(tmp_path):
    import yaml

    from kuaa.config import load_config
    from kuaa.errors import ConfigError

    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"models": {"frame_source": "bogus"}}), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(bad, project_root=".", ensure_dirs=False)
