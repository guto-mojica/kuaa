"""Library registry CRUD + film scanning tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from cinemateca.library import (
    delete_film,
    load_registry,
    register_film,
    save_registry,
)


def test_load_registry_missing_returns_empty(tmp_path: Path) -> None:
    """An absent films.json yields an empty registry, not a crash."""
    library_dir = tmp_path / "library"
    library_dir.mkdir()
    assert load_registry(library_dir) == {}


def test_save_then_load_registry_roundtrip(tmp_path: Path) -> None:
    """save_registry persists what load_registry reads back."""
    library_dir = tmp_path / "library"
    library_dir.mkdir()
    payload = {
        "jeca_tatu": {
            "slug": "jeca_tatu",
            "title": "Jeca Tatu",
            "year": 1959,
            "raw_filename": "jeca_tatu.mp4",
            "added_at": "2026-05-20T10:00:00Z",
        }
    }
    save_registry(library_dir, payload)
    assert (library_dir / "films.json").exists()
    assert load_registry(library_dir) == payload


def test_register_film_adds_entry(tmp_path: Path) -> None:
    """register_film writes a new entry; duplicate slug raises."""
    library_dir = tmp_path / "library"
    library_dir.mkdir()
    register_film(
        library_dir,
        slug="jeca_tatu",
        title="Jeca Tatu",
        year=1959,
        raw_filename="jeca_tatu.mp4",
    )
    reg = load_registry(library_dir)
    assert reg["jeca_tatu"]["title"] == "Jeca Tatu"
    assert reg["jeca_tatu"]["year"] == 1959
    assert "added_at" in reg["jeca_tatu"]


def test_register_film_duplicate_slug_raises(tmp_path: Path) -> None:
    library_dir = tmp_path / "library"
    library_dir.mkdir()
    register_film(library_dir, slug="x", title="X", year=2000, raw_filename="x.mp4")
    with pytest.raises(ValueError, match="already registered"):
        register_film(library_dir, slug="x", title="X2", year=2001, raw_filename="x.mp4")


def test_delete_film_removes_entry(tmp_path: Path) -> None:
    library_dir = tmp_path / "library"
    library_dir.mkdir()
    register_film(library_dir, slug="x", title="X", year=2000, raw_filename="x.mp4")
    delete_film(library_dir, slug="x")
    assert load_registry(library_dir) == {}


def test_delete_film_unknown_slug_raises(tmp_path: Path) -> None:
    library_dir = tmp_path / "library"
    library_dir.mkdir()
    with pytest.raises(KeyError, match="not in registry"):
        delete_film(library_dir, slug="ghost")
