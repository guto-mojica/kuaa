"""A6: FilmContext is provided by one dependency; routes don't construct it inline."""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ROUTE_FILES = sorted((REPO / "api" / "routes").glob("*.py"))


def test_no_route_body_constructs_filmcontext() -> None:
    offenders: list[str] = []
    for py in ROUTE_FILES:
        tree = ast.parse(py.read_text(), filename=str(py))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in {"for_film", "from_config"}:
                val = node.value
                if isinstance(val, ast.Name) and val.id == "FilmContext":
                    offenders.append(f"{py.name}:{node.lineno} FilmContext.{node.attr}")
    assert not offenders, "routes must use the FilmContext dependency:\n" + "\n".join(offenders)


def test_optional_dependency_aggregate_and_per_film(client, seed_metadata) -> None:
    seed_metadata()
    # "default" is the slug the seed_metadata factory always creates.
    slug = "default"
    # per-film resolves; aggregate (no ?film=) yields the aggregate page — both 200.
    assert client.get(f"/tab/search?film={slug}").status_code == 200
    assert client.get("/tab/search").status_code == 200
