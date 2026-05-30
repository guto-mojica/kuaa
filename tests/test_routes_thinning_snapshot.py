"""Byte-identical gate for the route-thinning relocations (A2)."""
from __future__ import annotations

import pytest

from tests._snapshot import assert_snapshot


@pytest.fixture()
def seeded_client(seed_metadata, client):
    seed_metadata()
    return client


CASES = [
    ("search_noindex", "/api/search?q=river&retriever=clip"),
    ("tab_search", "/tab/search"),
    ("library_tree", "/api/library/tree"),
    ("library_filter", "/api/library/filter"),
    ("tab_annotate", "/tab/annotate"),
    ("tab_scenes", "/tab/scenes"),
    ("tab_rimas", "/tab/rimas"),
]


@pytest.mark.parametrize("name,url", CASES)
def test_route_render_stable(seeded_client, name, url) -> None:
    r = seeded_client.get(url)
    assert r.status_code == 200
    assert_snapshot(f"routes_thinning/{name}", r.text)


def test_eval_admin_gate_returns_403(client) -> None:
    """/eval must return 403 when EVAL_ADMIN_TOKEN is unset (the default)."""
    assert client.get("/eval").status_code == 403
