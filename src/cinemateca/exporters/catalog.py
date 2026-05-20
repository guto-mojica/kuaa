"""Domain-aware catalog export assembly."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cinemateca.annotator import load as load_annotations
from cinemateca.domain import DomainPack, export_record, load_domain_from_config
from cinemateca.scene_ids import scene_id_key

CATALOG_EXPORT_SCHEMA_VERSION = "1.0"


class ExportError(RuntimeError):
    """Raised when a catalog cannot be exported from available artifacts."""


@dataclass(frozen=True)
class CatalogExportMeta:
    """Top-level export metadata for reload/provenance."""

    schema_version: str
    generated_at: str
    scene_count: int
    domain: dict[str, Any]
    artifacts: dict[str, str]
    missing_artifacts: list[str]


@dataclass(frozen=True)
class CatalogExport:
    """In-memory structured catalog export."""

    meta: CatalogExportMeta
    scenes: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "export": {
                "schema_version": self.meta.schema_version,
                "generated_at": self.meta.generated_at,
                "scene_count": self.meta.scene_count,
                "domain": self.meta.domain,
                "artifacts": self.meta.artifacts,
                "missing_artifacts": self.meta.missing_artifacts,
            },
            "scenes": self.scenes,
        }


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _require_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ExportError(f"Required catalog artifact is missing: {path}")
    data = _read_json(path, [])
    if not isinstance(data, list):
        raise ExportError(f"Catalog artifact must be a JSON list: {path}")
    return [item for item in data if isinstance(item, dict)]


def _dict_list_by_scene(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        scene_id_key(item["scene_id"]): item
        for item in items
        if "scene_id" in item
    }


def _tags_by_scene(tag_index: dict[str, Any]) -> dict[str, set[str]]:
    by_scene: dict[str, set[str]] = {}
    for tag, ids in tag_index.items():
        if not isinstance(ids, list | tuple | set):
            continue
        clean_tag = str(tag).strip()
        if not clean_tag:
            continue
        for raw_id in ids:
            by_scene.setdefault(scene_id_key(raw_id), set()).add(clean_tag)
    return by_scene


def _artifact_paths(cfg: Any) -> dict[str, Path]:
    metadata_dir = Path(cfg.paths.metadata_dir)
    embeddings_dir = Path(cfg.paths.embeddings_dir)
    return {
        "keyframes_metadata": metadata_dir / "keyframes_metadata.json",
        "scene_descriptions": metadata_dir / cfg.llm.descriptions_filename,
        "scene_tags": metadata_dir / cfg.llm.tags_filename,
        "visual_analysis": metadata_dir / "visual_analysis.json",
        "manual_annotations": metadata_dir / "manual_annotations.json",
        "embeddings": embeddings_dir / cfg.embeddings.filename,
        "embeddings_mapping": embeddings_dir / cfg.embeddings.mapping_filename,
        "run_manifest": metadata_dir / "run_manifest.json",
    }


def _stringify_path(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def _full_scene_record(
    keyframe: dict[str, Any],
    description: dict[str, Any],
    visual: dict[str, Any],
    generated_tags: set[str],
    manual_tags: list[str],
) -> dict[str, Any]:
    record: dict[str, Any] = {}
    record.update(keyframe)
    record.update(description)

    if "filepath" in keyframe and "keyframe_path" not in record:
        record["keyframe_path"] = str(keyframe.get("filepath") or "")
    if "keyframe_path" in record:
        record["keyframe_path"] = str(record.get("keyframe_path") or "")

    manual_clean = [str(tag) for tag in manual_tags if str(tag).strip()]
    record["generated_tags"] = sorted(generated_tags)
    record["manual_tags"] = manual_clean
    record["tags"] = sorted(set(record.get("tags") or []) | generated_tags | set(manual_clean))
    record["visual_analysis"] = visual
    return record


def build_catalog_export(
    cfg: Any,
    *,
    generated_at: datetime | None = None,
    domain_pack: DomainPack | None = None,
) -> CatalogExport:
    """Build a reloadable domain-shaped catalog export from metadata artifacts."""

    pack = domain_pack or load_domain_from_config(cfg)
    paths = _artifact_paths(cfg)

    keyframes = _require_json_list(paths["keyframes_metadata"])
    descriptions = _read_json(paths["scene_descriptions"], [])
    visual_analysis = _read_json(paths["visual_analysis"], [])
    scene_tags = _read_json(paths["scene_tags"], {})
    annotations = load_annotations(Path(cfg.paths.metadata_dir))

    if not isinstance(descriptions, list):
        raise ExportError(f"Catalog artifact must be a JSON list: {paths['scene_descriptions']}")
    if not isinstance(visual_analysis, list):
        raise ExportError(f"Catalog artifact must be a JSON list: {paths['visual_analysis']}")
    if not isinstance(scene_tags, dict):
        raise ExportError(f"Catalog artifact must be a JSON object: {paths['scene_tags']}")

    desc_by_scene = _dict_list_by_scene(
        [item for item in descriptions if isinstance(item, dict)]
    )
    visual_by_scene = _dict_list_by_scene(
        [item for item in visual_analysis if isinstance(item, dict)]
    )
    tags_by_scene = _tags_by_scene(scene_tags)

    scenes: list[dict[str, Any]] = []
    for keyframe in keyframes:
        sid = scene_id_key(keyframe.get("scene_id", ""))
        full = _full_scene_record(
            keyframe,
            desc_by_scene.get(sid, {}),
            visual_by_scene.get(sid, {}),
            tags_by_scene.get(sid, set()),
            annotations.get(sid, []),
        )
        scenes.append(export_record(full, pack))

    seen_artifacts = {
        name: _stringify_path(path)
        for name, path in paths.items()
        if path.exists()
    }
    missing_artifacts = [
        name for name, path in paths.items() if not path.exists()
    ]
    now = generated_at or datetime.now(UTC)

    return CatalogExport(
        meta=CatalogExportMeta(
            schema_version=CATALOG_EXPORT_SCHEMA_VERSION,
            generated_at=now.isoformat().replace("+00:00", "Z"),
            scene_count=len(scenes),
            domain={
                "id": pack.id,
                "label": pack.label,
                "path": _stringify_path(pack.path) if pack.path else None,
            },
            artifacts=seen_artifacts,
            missing_artifacts=missing_artifacts,
        ),
        scenes=scenes,
    )


def catalog_export_to_json(export: CatalogExport) -> str:
    """Serialize a catalog export as stable, indented UTF-8 JSON text."""

    return json.dumps(export.to_dict(), indent=2, ensure_ascii=False)


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, int | float | bool):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def catalog_export_to_csv(export: CatalogExport) -> str:
    """Serialize scene records as flat CSV using JSON strings for containers."""

    fieldnames: list[str] = []
    for scene in export.scenes:
        for key in scene:
            if key not in fieldnames:
                fieldnames.append(key)

    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    writer.writeheader()
    for scene in export.scenes:
        writer.writerow({key: _csv_value(scene.get(key)) for key in fieldnames})
    return out.getvalue()
