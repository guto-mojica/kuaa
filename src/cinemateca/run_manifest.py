"""Run manifest generation for processing provenance."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_FILENAME = "run_manifest.json"
MANIFEST_SCHEMA_VERSION = "1.0"


def _coerce_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if hasattr(value, "to_dict"):
        return _coerce_jsonable(value.to_dict())
    if isinstance(value, Mapping):
        return {str(k): _coerce_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_coerce_jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted(_coerce_jsonable(v) for v in value)
    return value


def config_snapshot(cfg: Any) -> dict[str, Any]:
    """Return a stable JSON-compatible snapshot of the loaded config."""

    if hasattr(cfg, "to_dict"):
        raw = cfg.to_dict()
    else:
        raw = vars(cfg)
    return _coerce_jsonable(raw)


def config_hash(cfg: Any) -> str:
    """Return a SHA-256 hash of the loaded config snapshot."""

    payload = json.dumps(
        config_snapshot(cfg),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _iso_from_epoch(value: float | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value, timezone.utc).isoformat().replace("+00:00", "Z")


def input_identity(video_path: str | Path) -> dict[str, Any]:
    """Return a lightweight identity record for the input video path."""

    path = Path(video_path)
    record: dict[str, Any] = {
        "path": str(path),
        "resolved_path": str(path.resolve()) if path.exists() else None,
        "name": path.name,
        "exists": path.exists(),
    }
    if path.exists():
        stat = path.stat()
        record.update(
            {
                "size_bytes": stat.st_size,
                "mtime": _iso_from_epoch(stat.st_mtime),
            }
        )
    return record


def _get_path(root: Any, *parts: str, default: Any = None) -> Any:
    current = root
    for part in parts:
        if current is None:
            return default
        if isinstance(current, Mapping):
            current = current.get(part, default)
        else:
            current = getattr(current, part, default)
    return current


def model_snapshot(cfg: Any) -> dict[str, Any]:
    """Return configured model backend names and model revision hints."""

    return {
        "backends": _coerce_jsonable(_get_path(cfg, "models", default={})),
        "llm": {
            "model_id": _get_path(cfg, "llm", "model_id"),
            "revision": _get_path(cfg, "llm", "revision"),
            "scene_describer": _get_path(cfg, "models", "scene_describer"),
        },
        "embeddings": {
            "model": _get_path(cfg, "embeddings", "model"),
            "pretrained": _get_path(cfg, "embeddings", "pretrained"),
            "backend": _get_path(cfg, "models", "image_embedder"),
        },
        "visual_analysis": {
            "face_detector": _get_path(cfg, "models", "face_detector"),
            "object_detector": _get_path(cfg, "models", "object_detector"),
            "object_model": _get_path(
                cfg, "visual_analysis", "object_detection", "model"
            ),
            "environment_classifier": _get_path(
                cfg, "models", "environment_classifier"
            ),
        },
    }


def domain_snapshot(cfg: Any) -> dict[str, Any]:
    """Return selected domain pack identity without failing manifest writes."""

    try:
        from cinemateca.domain import load_domain_from_config

        pack = load_domain_from_config(cfg)
    except Exception as exc:  # noqa: BLE001 - manifest should not break processing
        return {"error": str(exc)}
    return {
        "id": pack.id,
        "label": pack.label,
        "path": str(pack.path.resolve()) if pack.path else None,
    }


def output_artifacts(cfg: Any) -> dict[str, dict[str, Any]]:
    """Return expected output artifact paths and existence state."""

    metadata_dir = Path(_get_path(cfg, "paths", "metadata_dir", default="."))
    embeddings_dir = Path(_get_path(cfg, "paths", "embeddings_dir", default="."))
    llm_desc = _get_path(
        cfg, "llm", "descriptions_filename", default="scene_descriptions.json"
    )
    llm_tags = _get_path(cfg, "llm", "tags_filename", default="scene_tags.json")
    emb_file = _get_path(
        cfg, "embeddings", "filename", default="keyframe_embeddings.npy"
    )
    mapping_file = _get_path(
        cfg, "embeddings", "mapping_filename", default="index_mapping.json"
    )

    paths = {
        "video_properties": metadata_dir / "video_properties.json",
        "keyframes_metadata": metadata_dir / "keyframes_metadata.json",
        "visual_analysis": metadata_dir / "visual_analysis.json",
        "scene_descriptions": metadata_dir / llm_desc,
        "scene_tags": metadata_dir / llm_tags,
        "manual_annotations": metadata_dir / "manual_annotations.json",
        "embeddings": embeddings_dir / emb_file,
        "embeddings_mapping": embeddings_dir / mapping_file,
        "run_manifest": metadata_dir / MANIFEST_FILENAME,
    }
    return {
        name: {"path": str(path), "exists": path.exists()}
        for name, path in paths.items()
    }


def _state_from_step_result(step: Any) -> str:
    if getattr(step, "skipped", False):
        return "skipped"
    return "done" if getattr(step, "success", False) else "error"


def steps_snapshot(result: Any | None) -> list[dict[str, Any]]:
    """Normalize PipelineResult or StepResults step records."""

    if result is None:
        return []
    if hasattr(result, "runs"):
        runs = getattr(result, "runs")
        return [
            {
                "name": run.name,
                "state": run.state,
                "duration_s": run.duration_s,
                "error": run.error,
                "output": _coerce_jsonable(run.output),
            }
            for run in runs
        ]
    if hasattr(result, "steps"):
        steps = getattr(result, "steps")
        return [
            {
                "name": step.name,
                "state": _state_from_step_result(step),
                "duration_s": step.duration_s,
                "error": step.error,
                "output": _coerce_jsonable(step.output),
            }
            for step in steps
        ]
    return []


def infer_status(result: Any | None, explicit_status: str | None = None) -> str:
    if explicit_status:
        return explicit_status
    if result is None:
        return "unknown"
    if hasattr(result, "ok"):
        return "done" if result.ok else "error"
    if hasattr(result, "success"):
        return "done" if result.success else "error"
    return "unknown"


def build_run_manifest(
    cfg: Any,
    video_path: str | Path,
    result: Any | None = None,
    *,
    status: str | None = None,
    started_at_epoch: float | None = None,
    finished_at_epoch: float | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Build the run manifest payload without writing it."""

    finished = finished_at_epoch or time.time()
    started = started_at_epoch
    total_duration_s = getattr(result, "total_duration_s", None)
    if started is None and total_duration_s is not None:
        started = finished - float(total_duration_s)

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run": {
            "status": infer_status(result, status),
            "started_at": _iso_from_epoch(started),
            "finished_at": _iso_from_epoch(finished),
            "total_duration_s": total_duration_s,
            "error": error,
        },
        "input": input_identity(video_path),
        "config": {
            "sha256": config_hash(cfg),
            "snapshot": config_snapshot(cfg),
        },
        "domain": domain_snapshot(cfg),
        "models": model_snapshot(cfg),
        "steps": steps_snapshot(result),
        "artifacts": output_artifacts(cfg),
    }


def write_manifest(path: str | Path, payload: dict[str, Any]) -> Path:
    """Atomically write a manifest payload to disk."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return target


def write_run_manifest(
    cfg: Any,
    video_path: str | Path,
    result: Any | None = None,
    *,
    status: str | None = None,
    started_at_epoch: float | None = None,
    finished_at_epoch: float | None = None,
    error: str | None = None,
) -> Path:
    """Build and write `run_manifest.json` beside metadata artifacts."""

    manifest_path = Path(cfg.paths.metadata_dir) / MANIFEST_FILENAME
    payload = build_run_manifest(
        cfg,
        video_path,
        result,
        status=status,
        started_at_epoch=started_at_epoch,
        finished_at_epoch=finished_at_epoch,
        error=error,
    )
    payload["artifacts"]["run_manifest"]["exists"] = True
    return write_manifest(manifest_path, payload)
