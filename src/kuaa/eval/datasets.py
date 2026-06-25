"""YAML query dataset loading and validation for retrieval evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from kuaa.errors import UserInputError
from kuaa.scene_ids import scene_id_key


class DatasetError(UserInputError):
    """Raised when an evaluation dataset is malformed."""

    default_code = "eval.dataset_invalid"


@dataclass(frozen=True)
class QueryCase:
    """One human-reviewed retrieval query."""

    id: str
    text: str
    relevant_scene_ids: tuple[str, ...]
    relevance: dict[str, float] = field(default_factory=dict)
    negative_scene_ids: tuple[str, ...] = ()
    intent: str = ""
    notes: str = ""


@dataclass(frozen=True)
class EvaluationDataset:
    """A versioned set of retrieval queries."""

    dataset: str
    version: int
    queries: tuple[QueryCase, ...]
    source: dict[str, Any] = field(default_factory=dict)
    label_status: str = ""
    path: Path | None = None


def _as_scene_ids(value: Any, *, field_name: str, query_id: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise DatasetError(f"{query_id}.{field_name} must be a list")
    out = tuple(scene_id_key(v) for v in value)
    if len(set(out)) != len(out):
        raise DatasetError(f"{query_id}.{field_name} contains duplicate ids")
    return out


def _as_relevance(value: Any, relevant: tuple[str, ...], *, query_id: str) -> dict[str, float]:
    if value is None:
        return {sid: 1.0 for sid in relevant}
    if not isinstance(value, dict):
        raise DatasetError(f"{query_id}.relevance must be a mapping of scene id to grade")

    grades: dict[str, float] = {}
    for raw_scene_id, raw_grade in value.items():
        sid = scene_id_key(raw_scene_id)
        try:
            grade = float(raw_grade)
        except (TypeError, ValueError) as exc:
            raise DatasetError(f"{query_id}.relevance[{raw_scene_id!r}] must be numeric") from exc
        if grade <= 0:
            raise DatasetError(f"{query_id}.relevance[{raw_scene_id!r}] must be positive")
        grades[sid] = grade

    for sid in relevant:
        grades.setdefault(sid, 1.0)
    return grades


def _load_query(raw: Any, *, index: int) -> QueryCase:
    if not isinstance(raw, dict):
        raise DatasetError(f"queries[{index}] must be a mapping")

    qid = str(raw.get("id") or "").strip()
    if not qid:
        raise DatasetError(f"queries[{index}].id is required")

    text = str(raw.get("text") or "").strip()
    if not text:
        raise DatasetError(f"{qid}.text is required")

    relevant = _as_scene_ids(
        raw.get("relevant_scene_ids"),
        field_name="relevant_scene_ids",
        query_id=qid,
    )
    if not relevant:
        raise DatasetError(f"{qid}.relevant_scene_ids must contain at least one id")

    negative = _as_scene_ids(
        raw.get("negative_scene_ids"),
        field_name="negative_scene_ids",
        query_id=qid,
    )
    relevance = _as_relevance(raw.get("relevance"), relevant, query_id=qid)

    return QueryCase(
        id=qid,
        text=text,
        relevant_scene_ids=relevant,
        relevance=relevance,
        negative_scene_ids=negative,
        intent=str(raw.get("intent") or "").strip(),
        notes=str(raw.get("notes") or "").strip(),
    )


def load_dataset(path: str | Path) -> EvaluationDataset:
    """Load and validate an evaluation YAML file."""

    dataset_path = Path(path)
    try:
        with open(dataset_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except FileNotFoundError as exc:
        raise DatasetError(f"Evaluation query file not found: {dataset_path}") from exc
    except yaml.YAMLError as exc:
        raise DatasetError(f"Evaluation query file is invalid YAML: {dataset_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise DatasetError("Evaluation query file must contain a mapping")

    name = str(raw.get("dataset") or "").strip()
    if not name:
        raise DatasetError("dataset is required")

    raw_version = raw.get("version")
    if raw_version is None:
        raise DatasetError("version is required")
    try:
        version = int(raw_version)
    except (TypeError, ValueError) as exc:
        raise DatasetError("version must be an integer") from exc

    raw_queries = raw.get("queries")
    if not isinstance(raw_queries, list) or not raw_queries:
        raise DatasetError("queries must be a non-empty list")

    queries = tuple(_load_query(q, index=i) for i, q in enumerate(raw_queries))
    ids = [q.id for q in queries]
    if len(set(ids)) != len(ids):
        raise DatasetError("queries contain duplicate ids")

    source = raw.get("source") or {}
    if not isinstance(source, dict):
        raise DatasetError("source must be a mapping when provided")

    return EvaluationDataset(
        dataset=name,
        version=version,
        queries=queries,
        source=source,
        label_status=str(raw.get("label_status") or "").strip(),
        path=dataset_path,
    )


def load_queries(root: Path, run_id: str) -> list[dict[str, Any]]:
    """Load the curated query list for the run. Empty when missing.

    Task 33 ships the seeded queries file; until then this returns
    ``[]`` and the /eval page renders an empty-state queue.
    """

    queries_path = root / f"{run_id}.queries.json"
    if not queries_path.exists():
        return []
    try:
        return json.loads(queries_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
