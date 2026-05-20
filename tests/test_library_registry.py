"""Library registry CRUD + film scanning tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cinemateca.library import load_registry, save_registry


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
