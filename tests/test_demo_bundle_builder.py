from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from scripts import build_demo_bundle


def _manifest(root: Path, filename: str = "demo.zip") -> dict:
    return {
        "version": "test-demo",
        "source": {
            "title": "Fixture film",
            "year": 1903,
        },
        "artifact_bundle": {
            "filename": filename,
            "url": "https://example.invalid/demo.zip",
            "sha256": None,
            "root": str(root),
        },
        "expected": {
            "min_keyframes": 2,
            "required_metadata": [
                "keyframes_metadata.json",
                "scene_descriptions.json",
                "scene_tags.json",
                "visual_analysis.json",
            ],
        },
        "checksums": {},
    }


def _write_manifest(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_runtime(root: Path) -> None:
    metadata = root / "metadata"
    frames = root / "frames" / "scenes" / "keyframes_content"
    embeddings = root / "embeddings"
    raw = root / "raw"
    for path in (metadata, frames, embeddings, raw):
        path.mkdir(parents=True, exist_ok=True)

    k1 = frames / "scene_001.jpg"
    k2 = frames / "scene_002.jpg"
    k1.write_bytes(b"jpeg-1")
    k2.write_bytes(b"jpeg-2")
    (raw / "demo.mp4").write_bytes(b"video")

    scenes = [
        {
            "scene_id": 1,
            "filepath": "frames/scenes/keyframes_content/scene_001.jpg",
            "start_time_s": 0.0,
        },
        {
            "scene_id": 2,
            "filepath": "frames/scenes/keyframes_content/scene_002.jpg",
            "start_time_s": 4.0,
        },
    ]
    (metadata / "keyframes_metadata.json").write_text(json.dumps(scenes), encoding="utf-8")
    (metadata / "scene_descriptions.json").write_text(
        json.dumps(
            [
                {"scene_id": 1, "description": "A train station."},
                {"scene_id": 2, "description": "A rail car interior."},
            ]
        ),
        encoding="utf-8",
    )
    (metadata / "scene_tags.json").write_text(
        json.dumps({"train": [1, 2], "interior": [2]}),
        encoding="utf-8",
    )
    (metadata / "visual_analysis.json").write_text(
        json.dumps(
            [
                {"scene_id": 1, "environment": {"location": "exterior"}},
                {"scene_id": 2, "environment": {"location": "interior"}},
            ]
        ),
        encoding="utf-8",
    )
    np.save(embeddings / "keyframe_embeddings.npy", np.ones((2, 3), dtype="float32"))
    (embeddings / "index_mapping.json").write_text(
        json.dumps(
            {
                "model": "CLIP ViT-B-32 (openai)",
                "dimension": 3,
                "total_vectors": 2,
                "normalized": True,
                "keyframe_paths": [
                    "frames/scenes/keyframes_content/scene_001.jpg",
                    "frames/scenes/keyframes_content/scene_002.jpg",
                ],
                "scene_ids": [1, 2],
            }
        ),
        encoding="utf-8",
    )


def _build(tmp_path: Path, *, include_raw: bool = True, update_manifest: bool = False):
    root = tmp_path / "runtime"
    _write_runtime(root)
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, _manifest(root))
    result = build_demo_bundle.build_demo_bundle(
        manifest_path=manifest_path,
        output_dir=tmp_path / "dist",
        include_raw=include_raw,
        update_manifest=update_manifest,
    )
    return result, manifest_path


def test_build_demo_bundle_writes_zip_and_release_artifacts(tmp_path):
    result, _manifest_path = _build(tmp_path)

    assert result.zip_path.exists()
    assert result.checksum_path.read_text(encoding="utf-8").startswith(result.bundle_sha256)
    assert result.manifest_preview_path.exists()
    assert result.release_snippet_path.exists()
    assert result.file_count == 9

    with zipfile.ZipFile(result.zip_path) as archive:
        names = archive.namelist()

    assert names == sorted(names)
    assert "metadata/keyframes_metadata.json" in names
    assert "frames/scenes/keyframes_content/scene_001.jpg" in names
    assert "embeddings/keyframe_embeddings.npy" in names
    assert "raw/demo.mp4" in names

    preview = json.loads(result.manifest_preview_path.read_text(encoding="utf-8"))
    assert preview["artifact_bundle"]["sha256"] == result.bundle_sha256
    assert preview["checksums"]["metadata/keyframes_metadata.json"]
    assert preview["checksums"]["raw/demo.mp4"]
    assert result.bundle_sha256 in result.release_snippet_path.read_text(encoding="utf-8")


def test_build_demo_bundle_is_deterministic_for_same_runtime(tmp_path):
    root = tmp_path / "runtime"
    _write_runtime(root)
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, _manifest(root, filename="deterministic.zip"))

    first = build_demo_bundle.build_demo_bundle(
        manifest_path=manifest_path,
        output_dir=tmp_path / "dist-a",
    )
    second = build_demo_bundle.build_demo_bundle(
        manifest_path=manifest_path,
        output_dir=tmp_path / "dist-b",
    )

    assert first.bundle_sha256 == second.bundle_sha256
    assert first.zip_path.read_bytes() == second.zip_path.read_bytes()


def test_build_demo_bundle_fails_when_runtime_validation_fails(tmp_path):
    root = tmp_path / "runtime"
    _write_runtime(root)
    (root / "metadata" / "scene_tags.json").unlink()
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, _manifest(root))

    with pytest.raises(build_demo_bundle.BundleBuildError, match="scene_tags.json"):
        build_demo_bundle.build_demo_bundle(
            manifest_path=manifest_path,
            output_dir=tmp_path / "dist",
        )

    assert not (tmp_path / "dist" / "demo.zip").exists()


def test_build_demo_bundle_can_exclude_raw_files(tmp_path):
    result, _manifest_path = _build(tmp_path, include_raw=False)

    with zipfile.ZipFile(result.zip_path) as archive:
        names = archive.namelist()

    assert "raw/demo.mp4" not in names
    assert "raw/demo.mp4" not in result.artifact_checksums
    assert result.file_count == 8


def test_build_demo_bundle_updates_manifest_only_when_requested(tmp_path):
    result, manifest_path = _build(tmp_path, update_manifest=True)

    updated = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert updated["artifact_bundle"]["sha256"] == result.bundle_sha256
    assert updated["checksums"] == result.artifact_checksums
