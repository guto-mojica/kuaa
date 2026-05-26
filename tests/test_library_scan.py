"""Unit tests for cinemateca.library.scan."""

from __future__ import annotations

import json
from pathlib import Path

from cinemateca.library import (
    Film,
    LibraryState,
    library_state,
    register_film,
    scan_library,
)


def test_film_dataclass_defaults():
    f = Film(slug="alpha", title="Alpha", raw_path=Path("/tmp/a.mp4"))
    assert f.scene_count == 0
    assert f.is_processed is False
    assert f.year is None


def test_scan_library_missing_dir_returns_empty(tmp_path):
    assert scan_library(tmp_path / "does-not-exist") == []


def test_scan_library_empty_registry_returns_empty(tmp_path):
    assert scan_library(tmp_path) == []


def test_scan_library_unprocessed_film_has_zero_scenes(tmp_path):
    register_film(tmp_path, slug="alpha", title="Alpha", year=2026, raw_filename="alpha.mp4")
    films = scan_library(tmp_path)
    assert len(films) == 1
    assert films[0].slug == "alpha"
    assert films[0].scene_count == 0
    assert films[0].is_processed is False


def test_scan_library_processed_film_counts_scenes(tmp_path):
    register_film(tmp_path, slug="alpha", title="Alpha", year=2026, raw_filename="alpha.mp4")
    metadata_dir = tmp_path / "alpha" / "metadata"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "keyframes_metadata.json").write_text(
        json.dumps([{"scene_id": 1}, {"scene_id": 2}, {"scene_id": 3}])
    )
    films = scan_library(tmp_path)
    assert films[0].scene_count == 3
    assert films[0].is_processed is True


def test_library_state_empty_returns_all_false(tmp_path):
    s = library_state(tmp_path)
    assert s == LibraryState(raw_present=False, index_present=False, scene_count=0)


def test_library_state_sums_scenes_across_films(tmp_path):
    for slug, n_scenes in [("alpha", 2), ("beta", 3)]:
        register_film(
            tmp_path, slug=slug, title=slug.title(), year=2026, raw_filename=f"{slug}.mp4"
        )
        (tmp_path / slug / "metadata").mkdir(parents=True)
        (tmp_path / slug / "metadata" / "keyframes_metadata.json").write_text(
            json.dumps([{"scene_id": i} for i in range(n_scenes)])
        )
    s = library_state(tmp_path)
    assert s.scene_count == 5
    assert s.is_processed is True
