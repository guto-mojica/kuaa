"""Byte-identical snapshot gate for the scenes_service decomposition (A1)."""

from __future__ import annotations

import pytest

from tests._snapshot import assert_snapshot


@pytest.fixture()
def seeded_client(seed_metadata, client):
    seed_metadata()  # writes the historical default dataset into temp dirs
    return client


def test_tab_scenes_snapshot(seeded_client) -> None:
    r = seeded_client.get("/tab/scenes")
    assert r.status_code == 200
    assert_snapshot("scenes_service/tab_scenes", r.text)


def test_api_scenes_grid_snapshot(seeded_client) -> None:
    r = seeded_client.get("/api/scenes?q=&group=tipo&sort=duration")
    assert r.status_code == 200
    assert_snapshot("scenes_service/api_scenes_grid_tipo_duration", r.text)


def test_scene_inspector_snapshot(seeded_client) -> None:
    # scene_id 351 is written by the default seed_metadata dataset (slug="default").
    r = seeded_client.get("/api/scenes/351/inspector?film=default&tab=properties&kind=cenas")
    assert r.status_code == 200
    assert_snapshot("scenes_service/inspector_cenas", r.text)
