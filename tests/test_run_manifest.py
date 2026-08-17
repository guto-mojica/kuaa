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


# ─────────────────────────────────────────────────────────────────────────────
# 0.10 — skip_existing must read the hash the manifest already writes
# ─────────────────────────────────────────────────────────────────────────────


def test_step_hash_moves_only_for_the_step_whose_config_changed(tmp_config):
    """Scoped per step, not over the whole config.

    Hashing everything would make an unrelated edit — a retrieval weight, an
    output path — invalidate every LLM description in the library and silently
    trigger hours of re-inference on the next run. A step re-runs only when
    something that actually feeds it moved.
    """
    from kuaa.run_manifest import step_config_hash

    before = {
        step: step_config_hash(tmp_config, step)
        for step in ("scene_detection", "embeddings", "llm_description")
    }
    tmp_config.embeddings.batch_size = 99
    after = {step: step_config_hash(tmp_config, step) for step in before}

    assert after["embeddings"] != before["embeddings"]
    assert after["scene_detection"] == before["scene_detection"]
    assert after["llm_description"] == before["llm_description"]


def test_step_hash_is_none_for_an_undeclared_step(tmp_config):
    from kuaa.run_manifest import step_config_hash

    assert step_config_hash(tmp_config, "no_such_step") is None


def test_manifest_records_a_hash_per_step(tmp_config, tmp_path):
    from kuaa.run_manifest import STEP_CONFIG_SECTIONS

    payload = build_run_manifest(tmp_config, tmp_path / "v.mp4")
    stored = payload["config"]["step_sha256"]
    assert set(stored) == set(STEP_CONFIG_SECTIONS)
    assert all(isinstance(v, str) and v for v in stored.values())


def test_read_step_hashes_round_trips_through_the_written_manifest(tmp_config, tmp_path):
    """The manifest has recorded a config hash on every run and nothing has
    ever read one back. This is the read that turns it into a check."""
    from kuaa.run_manifest import read_step_hashes

    metadata_dir = tmp_path / "metadata"
    write_run_manifest(tmp_config, tmp_path / "v.mp4", metadata_dir=metadata_dir)

    stored = read_step_hashes(metadata_dir)
    assert stored["embeddings"]
    assert (
        stored
        == json.loads((metadata_dir / MANIFEST_FILENAME).read_text())["config"]["step_sha256"]
    )


def test_read_step_hashes_is_empty_when_there_is_no_manifest(tmp_path):
    from kuaa.run_manifest import read_step_hashes

    assert read_step_hashes(tmp_path / "nothing") == {}


def test_pipeline_refuses_to_skip_when_the_steps_config_moved(tmp_config, tmp_path, caplog):
    """A bare ``path.exists()`` reuses an artifact from a different generation.

    A library assembled that way is a mixed corpus with nothing on disk saying
    so — and the grades collected against it are spent on it.
    """
    from kuaa.pipeline import CatalogPipeline

    metadata_dir = tmp_path / "metadata"
    write_run_manifest(tmp_config, tmp_path / "v.mp4", metadata_dir=metadata_dir)

    pipeline = CatalogPipeline(tmp_config)
    pipeline._metadata_dir = lambda: metadata_dir  # type: ignore[method-assign]

    assert pipeline._can_skip("embeddings", exists=True), "unchanged config still skips"

    tmp_config.embeddings.batch_size = 99
    with caplog.at_level("WARNING"):
        assert not pipeline._can_skip("embeddings", exists=True)
    assert "config changed" in caplog.text
    # ...and a step the edit does not feed is still reused.
    assert pipeline._can_skip("scene_detection", exists=True)


def test_pipeline_skips_a_film_processed_before_provenance_checking(tmp_config, tmp_path):
    """No recorded hash → skip as before. The guard cannot retroactively know
    what produced that file, and re-running every pre-existing library would be
    a worse answer than saying so in the log."""
    from kuaa.pipeline import CatalogPipeline

    pipeline = CatalogPipeline(tmp_config)
    pipeline._metadata_dir = lambda: tmp_path / "no_manifest_here"  # type: ignore[method-assign]
    assert pipeline._can_skip("embeddings", exists=True)
