from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from io import StringIO

import pytest

from cinemateca.exporters import (
    ExportError,
    build_catalog_export,
    catalog_export_to_csv,
    catalog_export_to_json,
)


def test_catalog_export_requires_keyframes_metadata(tmp_config):
    with pytest.raises(ExportError, match="Required catalog artifact"):
        build_catalog_export(tmp_config)


def test_catalog_export_builds_domain_shaped_json(tmp_config, seed_metadata):
    seed_metadata()

    export = build_catalog_export(
        tmp_config,
        generated_at=datetime(2026, 5, 20, 12, 0, tzinfo=UTC),
    )
    data = json.loads(catalog_export_to_json(export))

    assert data["export"]["schema_version"] == "1.0"
    assert data["export"]["generated_at"] == "2026-05-20T12:00:00Z"
    assert data["export"]["scene_count"] == 2
    assert data["export"]["domain"]["id"] == "archive"
    assert "keyframes_metadata" in data["export"]["artifacts"]
    assert "embeddings" in data["export"]["missing_artifacts"]

    first = data["scenes"][0]
    assert first["scene_id"] == 351
    assert first["description"] == "a man walking outdoors at dawn"
    assert "exterior" in first["tags"]
    assert "dia" in first["tags"]
    assert "manual-only" not in first["tags"]

    second = data["scenes"][1]
    assert "manual-only" in second["tags"]


def test_catalog_export_csv_flattens_lists(tmp_config, seed_metadata):
    seed_metadata()

    export = build_catalog_export(tmp_config)
    rows = list(csv.DictReader(StringIO(catalog_export_to_csv(export))))

    assert len(rows) == 2
    assert rows[0]["scene_id"] == "351"
    assert json.loads(rows[0]["tags"]) == ["dia", "exterior"]


def test_export_routes_return_json_and_csv(client, seed_metadata):
    seed_metadata()

    json_response = client.get("/api/export/catalog.json")
    assert json_response.status_code == 200
    assert json_response.headers["content-type"].startswith("application/json")
    assert json_response.json()["export"]["domain"]["id"] == "archive"

    csv_response = client.get("/api/export/catalog.csv")
    assert csv_response.status_code == 200
    assert csv_response.headers["content-type"].startswith("text/csv")
    assert "scene_id,keyframe_id" in csv_response.text


def test_export_route_missing_artifact_is_clear(client):
    response = client.get("/api/export/catalog.json")

    assert response.status_code == 404
    assert "Required catalog artifact is missing" in response.json()["detail"]
