"""Guard against resurrecting the dead api/services migration scaffolding."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DEAD = [
    "api/services/_scenes_list.py",
    "api/services/_scene_detail.py",
    "api/services/_search_text.py",
    "api/services/_search_image.py",
    "api/services/film_service.py",
    "api/services/dtos.py",
]
DEAD_MODNAMES = {
    "api.services._scenes_list",
    "api.services._scene_detail",
    "api.services._search_text",
    "api.services._search_image",
    "api.services.film_service",
    "api.services.dtos",
}


@pytest.mark.parametrize("rel", DEAD)
def test_dead_module_is_deleted(rel: str) -> None:
    assert not (REPO / rel).exists(), f"{rel} should be deleted (dead code)"


def test_no_module_imports_dead_scaffolding() -> None:
    """Walk every .py under api/, tests/, src/ and assert none import the dead names."""
    offenders: list[str] = []
    roots = [REPO / "api", REPO / "src", REPO / "tests"]
    for root in roots:
        for py in root.rglob("*.py"):
            if py.name == "test_no_dead_service_modules.py":
                continue
            tree = ast.parse(py.read_text(), filename=str(py))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module in DEAD_MODNAMES:
                    offenders.append(f"{py}: from {node.module}")
                if isinstance(node, ast.Import):
                    for a in node.names:
                        if a.name in DEAD_MODNAMES:
                            offenders.append(f"{py}: import {a.name}")
    assert not offenders, "dead-module imports found:\n" + "\n".join(offenders)
