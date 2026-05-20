from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import api.deps as deps
from scripts import prepare_demo

_PATH_NAMES = (
    "data_dir",
    "raw_dir",
    "frames_dir",
    "metadata_dir",
    "embeddings_dir",
    "models_dir",
    "outputs_dir",
    "logs_dir",
)


def _write_config(path: Path, root: Path, marker: str) -> None:
    lines = [
        "project:",
        f'  name: "{marker}"',
        "paths:",
    ]
    for name in _PATH_NAMES:
        lines.append(f'  {name}: "{root / name}"')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def test_get_config_prefers_explicit_env_config(tmp_path, monkeypatch):
    config_path = tmp_path / "explicit.yaml"
    _write_config(config_path, tmp_path / "explicit", "Explicit demo config")

    monkeypatch.setenv(deps.CONFIG_ENV_VAR, str(config_path))
    deps.get_config.cache_clear()
    try:
        cfg = deps.get_config()
        assert cfg.project.name == "Explicit demo config"
        assert cfg.paths.metadata_dir == tmp_path / "explicit" / "metadata_dir"
    finally:
        deps.get_config.cache_clear()


def test_selected_config_path_falls_back_to_local_yaml(tmp_path, monkeypatch):
    local = tmp_path / "config" / "local.yaml"
    local.parent.mkdir()
    local.write_text("project:\n  name: Local\n", encoding="utf-8")

    monkeypatch.delenv(deps.CONFIG_ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)

    assert deps.selected_config_path() == Path("config/local.yaml")



def _demo_manifest(min_keyframes: int = 2) -> dict:
    return {
        "expected": {
            "min_keyframes": min_keyframes,
            "required_metadata": [
                "keyframes_metadata.json",
                "scene_descriptions.json",
                "scene_tags.json",
                "visual_analysis.json",
            ],
        },
        "checksums": {},
    }


def _write_demo_runtime(
    root: Path,
    *,
    embedding_rows: int = 2,
    total_vectors: int = 2,
) -> dict[str, Path]:
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
    (raw / "demo.mp4").write_bytes(b"")

    scenes = [
        {"scene_id": 1, "filepath": str(k1), "start_time_s": 0.0},
        {"scene_id": 2, "filepath": str(k2), "start_time_s": 12.0},
    ]
    (metadata / "keyframes_metadata.json").write_text(json.dumps(scenes), encoding="utf-8")
    (metadata / "scene_descriptions.json").write_text(
        json.dumps([
            {"scene_id": 1, "description": "Train station exterior."},
            {"scene_id": 2, "description": "Bandits enter the rail car."},
        ]),
        encoding="utf-8",
    )
    (metadata / "scene_tags.json").write_text(
        json.dumps({"train": [1, 2], "exterior": [1]}),
        encoding="utf-8",
    )
    (metadata / "visual_analysis.json").write_text(
        json.dumps([
            {"scene_id": 1, "environment": {"location": "exterior"}},
            {"scene_id": 2, "environment": {"location": "interior"}},
        ]),
        encoding="utf-8",
    )

    np.save(embeddings / "keyframe_embeddings.npy", np.ones((embedding_rows, 3), dtype="float32"))
    (embeddings / "index_mapping.json").write_text(
        json.dumps({
            "model": "CLIP ViT-B-32 (openai)",
            "dimension": 3,
            "total_vectors": total_vectors,
            "normalized": True,
            "keyframe_paths": [str(k1), str(k2)],
            "scene_ids": [1, 2],
        }),
        encoding="utf-8",
    )
    return {"metadata": metadata, "frames": frames, "embeddings": embeddings}


def test_prepare_demo_check_passes_tiny_fixture(tmp_path):
    root = tmp_path / "runtime"
    _write_demo_runtime(root)

    result = prepare_demo.check_demo(root, _demo_manifest())

    assert result.ok
    assert result.scene_count == 2
    assert result.keyframe_count == 2
    assert result.embedding_count == 2


def test_prepare_demo_reports_missing_required_artifact(tmp_path):
    root = tmp_path / "runtime"
    paths = _write_demo_runtime(root)
    (paths["metadata"] / "scene_tags.json").unlink()

    result = prepare_demo.check_demo(root, _demo_manifest())

    assert not result.ok
    assert any("scene_tags.json" in error for error in result.errors)


def test_prepare_demo_reports_invalid_json(tmp_path):
    root = tmp_path / "runtime"
    paths = _write_demo_runtime(root)
    (paths["metadata"] / "scene_descriptions.json").write_text("{", encoding="utf-8")

    result = prepare_demo.check_demo(root, _demo_manifest())

    assert not result.ok
    assert any("Invalid JSON" in error for error in result.errors)


def test_prepare_demo_reports_embedding_mapping_mismatch(tmp_path):
    root = tmp_path / "runtime"
    _write_demo_runtime(root, embedding_rows=1)

    result = prepare_demo.check_demo(root, _demo_manifest())

    assert not result.ok
    assert any("Embedding row count" in error for error in result.errors)


def test_prepare_demo_reports_checksum_mismatch(tmp_path):
    root = tmp_path / "runtime"
    _write_demo_runtime(root)
    manifest = _demo_manifest()
    manifest["checksums"] = {
        "metadata/keyframes_metadata.json": "0" * 64,
    }

    result = prepare_demo.check_demo(root, manifest)

    assert not result.ok
    assert any("Checksum mismatch" in error for error in result.errors)
