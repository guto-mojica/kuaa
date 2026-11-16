"""The inspector resolves one film without scanning the whole library."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import kuaa.library
import kuaa.library.scan
from api.services.scenes import build_inspector_context


def _write_film(library_dir: Path, slug: str, title: str, kf_text: str | None) -> None:
    meta = library_dir / slug / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    if kf_text is not None:
        (meta / "keyframes_metadata.json").write_text(kf_text, encoding="utf-8")


@pytest.fixture()
def two_films(tmp_config):
    lib = Path(tmp_config.paths.library_dir)
    lib.mkdir(parents=True, exist_ok=True)
    kf = json.dumps(
        [
            {
                "scene_id": 1,
                "filepath": str(lib / "alpha" / "frames" / "s1.jpg"),
                "start_time_s": 1.0,
                "end_time_s": 4.0,
            }
        ]
    )
    _write_film(lib, "alpha", "Alpha Film", kf)
    _write_film(lib, "beta", "Beta Film", "{ not json")
    (lib / "films.json").write_text(
        json.dumps(
            {
                "alpha": {"title": "Alpha Film", "year": 1931, "raw_filename": "a.mp4"},
                "beta": {"title": "Beta Film", "raw_filename": "b.mp4"},
            }
        ),
        encoding="utf-8",
    )
    return tmp_config


def _forbid_scan(monkeypatch) -> None:
    def boom(*_a, **_k):
        raise AssertionError("scan_library must not run for a single-film inspector")

    monkeypatch.setattr(kuaa.library, "scan_library", boom)
    monkeypatch.setattr(kuaa.library.scan, "scan_library", boom)


def test_inspector_does_not_scan_library(two_films, monkeypatch):
    _forbid_scan(monkeypatch)
    ctx = build_inspector_context(two_films, scene_id=1, slug="alpha")
    assert ctx is not None
    film = ctx["selected_film"]
    assert film.title == "Alpha Film"
    assert film.year == 1931
    assert film.slug == "alpha"
    assert film.scene_count == 1
    assert film.is_processed is True


def test_inspector_ignores_other_films_broken_metadata(two_films, monkeypatch):
    _forbid_scan(monkeypatch)
    assert build_inspector_context(two_films, scene_id=1, slug="alpha") is not None
