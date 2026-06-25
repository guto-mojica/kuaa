from __future__ import annotations

import json

from kuaa.pipeline import PipelineResult, StepResult, StepResults, StepRun
from kuaa.run_manifest import (
    MANIFEST_FILENAME,
    build_run_manifest,
    config_hash,
    input_identity,
    write_run_manifest,
)


def test_config_hash_changes_when_config_changes(tmp_config):
    before = config_hash(tmp_config)
    tmp_config.embeddings.batch_size = 99

    assert config_hash(tmp_config) != before


def test_input_identity_for_existing_file(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"abc")

    identity = input_identity(video)

    assert identity["name"] == "clip.mp4"
    assert identity["exists"] is True
    assert identity["size_bytes"] == 3
    assert identity["resolved_path"].endswith("clip.mp4")


def test_manifest_captures_pipeline_result_errors(tmp_config, tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"abc")
    result = PipelineResult(
        video_path=str(video),
        steps=[
            StepResult("scene_detection", success=False, duration_s=1.25, error="boom"),
            StepResult("embeddings", success=True, skipped=True),
        ],
        total_duration_s=1.25,
    )

    payload = build_run_manifest(
        tmp_config,
        video,
        result,
        started_at_epoch=1_779_292_800,
        finished_at_epoch=1_779_292_801,
    )

    assert payload["run"]["status"] == "error"
    assert payload["run"]["started_at"] == "2026-05-20T16:00:00Z"
    assert payload["input"]["size_bytes"] == 3
    assert payload["domain"]["id"] == "archive"
    assert payload["models"]["llm"]["revision"] == "2025-01-09"
    assert payload["steps"][0]["state"] == "error"
    assert payload["steps"][0]["error"] == "boom"
    assert payload["steps"][1]["state"] == "skipped"


def test_manifest_captures_step_results_blocked_state(tmp_config, tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"abc")
    result = StepResults(
        video_path=str(video),
        runs=[
            StepRun("visual_analysis", "blocked", error="missing keyframes"),
        ],
    )

    payload = build_run_manifest(
        tmp_config,
        video,
        result,
        status="error",
        error="missing keyframes",
    )

    assert payload["run"]["status"] == "error"
    assert payload["run"]["error"] == "missing keyframes"
    assert payload["steps"] == [
        {
            "name": "visual_analysis",
            "state": "blocked",
            "duration_s": 0.0,
            "error": "missing keyframes",
            "output": None,
        }
    ]


def test_write_run_manifest_writes_next_to_metadata(tmp_config, tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"abc")
    result = StepResults(
        video_path=str(video),
        runs=[StepRun("embeddings", "done", duration_s=0.5)],
    )

    path = write_run_manifest(tmp_config, video, result, started_at_epoch=1.0)

    assert path == tmp_config.paths.metadata_dir / MANIFEST_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["run"]["status"] == "done"
    assert payload["artifacts"]["run_manifest"]["path"].endswith(MANIFEST_FILENAME)
    assert payload["artifacts"]["run_manifest"]["exists"] is True
