"""Domain pack loading, validation, prompt selection, and export mapping."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from kuaa.config import Settings
from kuaa.errors import UserInputError

DEFAULT_DOMAIN_PACK = "archive"
DEFAULT_DOMAIN_PACKS_DIR = "config/domains"


class DomainError(UserInputError):
    """Raised when a domain pack is missing or malformed."""

    default_code = "domain.invalid"


@dataclass(frozen=True)
class MetadataField:
    """One domain metadata field definition."""

    name: str
    label: str
    type: str
    required: bool = False
    values: tuple[str, ...] = ()
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PromptTemplate:
    """One prompt template and token budget."""

    key: str
    prompt: str
    max_new_tokens: int


@dataclass(frozen=True)
class DomainPack:
    """Validated domain-pack configuration."""

    id: str
    label: str
    description: str
    metadata_fields: tuple[MetadataField, ...]
    prompt_templates: dict[str, PromptTemplate]
    taxonomy: dict[str, list[Any]]
    filters: tuple[dict[str, Any], ...]
    export_mapping: dict[str, str]
    sample_outputs: tuple[dict[str, Any], ...]
    evaluation: dict[str, Any]
    path: Path | None = None


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DomainError(f"{label} must be a mapping")
    return value


def _require_name(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise DomainError(f"{label} is required")
    return text


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except FileNotFoundError as exc:
        raise DomainError(f"Domain pack not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise DomainError(f"Domain pack is invalid YAML: {path}: {exc}") from exc
    return _require_mapping(raw, "domain pack")


def _load_fields(raw_fields: Any) -> tuple[MetadataField, ...]:
    if not isinstance(raw_fields, list) or not raw_fields:
        raise DomainError("metadata_fields must be a non-empty list")

    fields: list[MetadataField] = []
    names: set[str] = set()
    for idx, raw in enumerate(raw_fields):
        item = _require_mapping(raw, f"metadata_fields[{idx}]")
        name = _require_name(item.get("name"), f"metadata_fields[{idx}].name")
        if name in names:
            raise DomainError(f"duplicate metadata field: {name}")
        names.add(name)
        label = _require_name(item.get("label"), f"metadata_fields[{idx}].label")
        field_type = _require_name(item.get("type"), f"metadata_fields[{idx}].type")
        values = item.get("values") or ()
        if values and not isinstance(values, list):
            raise DomainError(f"metadata_fields[{idx}].values must be a list")
        extras = {
            k: v
            for k, v in item.items()
            if k not in {"name", "label", "type", "required", "values"}
        }
        fields.append(
            MetadataField(
                name=name,
                label=label,
                type=field_type,
                required=bool(item.get("required", False)),
                values=tuple(str(v) for v in values),
                extras=extras,
            )
        )
    return tuple(fields)


def _load_prompts(raw_prompts: Any) -> dict[str, PromptTemplate]:
    prompts_raw = _require_mapping(raw_prompts, "prompt_templates")
    if not prompts_raw:
        raise DomainError("prompt_templates must not be empty")

    prompts: dict[str, PromptTemplate] = {}
    for key, raw in prompts_raw.items():
        prompt_key = _require_name(key, "prompt_templates key")
        item = _require_mapping(raw, f"prompt_templates.{prompt_key}")
        prompt = _require_name(item.get("prompt"), f"prompt_templates.{prompt_key}.prompt")
        raw_max_new_tokens = item.get("max_new_tokens")
        if raw_max_new_tokens is None:
            raise DomainError(f"prompt_templates.{prompt_key}.max_new_tokens is required")
        try:
            max_new_tokens = int(raw_max_new_tokens)
        except (TypeError, ValueError) as exc:
            raise DomainError(
                f"prompt_templates.{prompt_key}.max_new_tokens must be an integer"
            ) from exc
        if max_new_tokens <= 0:
            raise DomainError(f"prompt_templates.{prompt_key}.max_new_tokens must be positive")
        prompts[prompt_key] = PromptTemplate(
            key=prompt_key,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
    return prompts


def _load_export_mapping(raw_mapping: Any) -> dict[str, str]:
    mapping = _require_mapping(raw_mapping, "export_mapping")
    if not mapping:
        raise DomainError("export_mapping must not be empty")
    out: dict[str, str] = {}
    for key, value in mapping.items():
        out[_require_name(key, "export_mapping key")] = _require_name(
            value, f"export_mapping.{key}"
        )
    return out


def load_domain_pack(path: str | Path) -> DomainPack:
    """Load and validate a domain pack YAML file."""

    pack_path = Path(path)
    raw = _load_yaml(pack_path)

    taxonomy = raw.get("taxonomy") or {}
    if not isinstance(taxonomy, dict):
        raise DomainError("taxonomy must be a mapping when provided")

    filters = raw.get("filters") or []
    if not isinstance(filters, list):
        raise DomainError("filters must be a list when provided")
    for idx, item in enumerate(filters):
        _require_mapping(item, f"filters[{idx}]")

    sample_outputs = raw.get("sample_outputs") or []
    if not isinstance(sample_outputs, list):
        raise DomainError("sample_outputs must be a list when provided")
    for idx, item in enumerate(sample_outputs):
        _require_mapping(item, f"sample_outputs[{idx}]")

    evaluation = raw.get("evaluation") or {}
    if not isinstance(evaluation, dict):
        raise DomainError("evaluation must be a mapping when provided")

    return DomainPack(
        id=_require_name(raw.get("id"), "id"),
        label=_require_name(raw.get("label"), "label"),
        description=str(raw.get("description") or "").strip(),
        metadata_fields=_load_fields(raw.get("metadata_fields")),
        prompt_templates=_load_prompts(raw.get("prompt_templates")),
        taxonomy=taxonomy,
        filters=tuple(dict(f) for f in filters),
        export_mapping=_load_export_mapping(raw.get("export_mapping")),
        sample_outputs=tuple(dict(s) for s in sample_outputs),
        evaluation=evaluation,
        path=pack_path,
    )


def resolve_domain_pack_path(cfg: Settings, project_root: str | Path | None = None) -> Path:
    """Resolve the selected domain pack path from a loaded config object."""

    root = Path(project_root) if project_root is not None else Path.cwd()
    domain_cfg = getattr(cfg, "domain", None)

    explicit = getattr(domain_cfg, "path", None) if domain_cfg is not None else None
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.is_absolute() else root / path

    pack = (
        getattr(domain_cfg, "pack", DEFAULT_DOMAIN_PACK)
        if domain_cfg is not None
        else DEFAULT_DOMAIN_PACK
    )
    packs_dir = (
        getattr(domain_cfg, "packs_dir", DEFAULT_DOMAIN_PACKS_DIR)
        if domain_cfg is not None
        else DEFAULT_DOMAIN_PACKS_DIR
    )
    base = Path(packs_dir).expanduser()
    if not base.is_absolute():
        base = root / base
    return base / f"{pack}.yaml"


def load_domain_from_config(
    cfg: Settings,
    project_root: str | Path | None = None,
) -> DomainPack:
    """Load the selected domain pack from an application config."""

    return load_domain_pack(resolve_domain_pack_path(cfg, project_root))


def prompt_dict(pack: DomainPack) -> dict[str, tuple[str, int]]:
    """Return the scene-describer prompt mapping for a domain pack."""

    return {
        key: (template.prompt, template.max_new_tokens)
        for key, template in pack.prompt_templates.items()
    }


def get_path_value(record: Mapping[str, Any], path: str, default: Any = None) -> Any:
    """Resolve a dot-path from a nested metadata record."""

    current: Any = record
    for part in path.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            return default
    return current


def export_record(record: Mapping[str, Any], pack: DomainPack) -> dict[str, Any]:
    """Map one metadata record into the domain-specific export shape."""

    return {
        output_name: get_path_value(record, source_path)
        for output_name, source_path in pack.export_mapping.items()
    }
