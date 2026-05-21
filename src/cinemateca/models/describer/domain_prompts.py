"""Domain-pack prompt adapter for scene describer backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cinemateca.domain import DomainPack, load_domain_from_config, prompt_dict
from cinemateca.models.describer._common import PROMPTS


def prompts_from_config(
    cfg: Any | None,
    project_root: str | Path | None = None,
) -> dict[str, tuple[str, int]]:
    """Return selected domain prompts, falling back to legacy prompts without cfg."""

    if cfg is None:
        return dict(PROMPTS)
    pack = load_domain_from_config(cfg, project_root)
    return prompt_dict(pack)


def domain_from_config(
    cfg: Any | None,
    project_root: str | Path | None = None,
) -> DomainPack | None:
    """Return the selected domain pack, or None when no config was supplied."""

    if cfg is None:
        return None
    return load_domain_from_config(cfg, project_root)
