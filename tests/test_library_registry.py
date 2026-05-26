"""Library registry CRUD + film scanning tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cinemateca.library import (
    delete_film,
    library_state,
    load_registry,
    register_film,
    save_registry,
    scan_library,
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
    """register_film persists all fields to films.json."""
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
    assert reg["jeca_tatu"]["slug"] == "jeca_tatu"
    assert reg["jeca_tatu"]["title"] == "Jeca Tatu"
    assert reg["jeca_tatu"]["year"] == 1959
    assert reg["jeca_tatu"]["raw_filename"] == "jeca_tatu.mp4"
    assert "added_at" in reg["jeca_tatu"]


def test_register_film_duplicate_slug_raises(tmp_path: Path) -> None:
    """Duplicate slug raises ValueError; original entry is preserved."""
    library_dir = tmp_path / "library"
    library_dir.mkdir()
    register_film(library_dir, slug="x", title="X", year=2000, raw_filename="x.mp4")
    with pytest.raises(ValueError, match="already registered"):
        register_film(library_dir, slug="x", title="X2", year=2001, raw_filename="x.mp4")
    reg = load_registry(library_dir)
    assert len(reg) == 1
    assert reg["x"]["title"] == "X"


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


def _make_film_layout(library_dir: Path, slug: str, scene_count: int) -> None:
    """Create a minimal per-film directory layout with N scenes."""
    film_dir = library_dir / slug
    (film_dir / "raw").mkdir(parents=True)
    (film_dir / "metadata").mkdir(parents=True)
    (film_dir / "embeddings").mkdir(parents=True)
    (film_dir / "raw" / f"{slug}.mp4").write_bytes(b"")
    if scene_count > 0:
        kf = [{"scene_id": i} for i in range(scene_count)]
        (film_dir / "metadata" / "keyframes_metadata.json").write_text(
            json.dumps(kf), encoding="utf-8"
        )


def test_scan_library_reads_registry_and_disk(tmp_path: Path) -> None:
    """scan_library returns one Film per registry entry with real disk state."""
    library_dir = tmp_path / "library"
    library_dir.mkdir()
    register_film(library_dir, slug="a", title="A", year=2000, raw_filename="a.mp4")
    register_film(library_dir, slug="b", title="B", year=2001, raw_filename="b.mp4")
    _make_film_layout(library_dir, "a", scene_count=5)
    _make_film_layout(library_dir, "b", scene_count=0)

    films = scan_library(library_dir)
    by_slug = {f.slug: f for f in films}
    assert set(by_slug) == {"a", "b"}
    assert by_slug["a"].scene_count == 5
    assert by_slug["a"].is_processed is True
    assert by_slug["b"].scene_count == 0
    assert by_slug["b"].is_processed is False
    assert by_slug["a"].title == "A"
    assert by_slug["a"].year == 2000
    assert by_slug["a"].raw_path == library_dir / "a" / "raw" / "a.mp4"


def test_scan_library_returns_empty_on_missing_dir(tmp_path: Path) -> None:
    assert scan_library(tmp_path / "library_does_not_exist") == []


def test_library_state_empty_library_returns_zero_state(tmp_path: Path) -> None:
    """Empty registry returns LibraryState(False, False, 0) without scanning."""
    library_dir = tmp_path / "library"
    library_dir.mkdir()
    from cinemateca.library import LibraryState

    state = library_state(library_dir)
    assert state == LibraryState(raw_present=False, index_present=False, scene_count=0)


def test_library_state_aggregates_across_films(tmp_path: Path) -> None:
    library_dir = tmp_path / "library"
    library_dir.mkdir()
    register_film(library_dir, slug="a", title="A", year=2000, raw_filename="a.mp4")
    register_film(library_dir, slug="b", title="B", year=2001, raw_filename="b.mp4")
    _make_film_layout(library_dir, "a", scene_count=5)
    _make_film_layout(library_dir, "b", scene_count=3)
    # Mark "a" indexed:
    (library_dir / "a" / "embeddings" / "keyframe_embeddings.npy").write_bytes(b"x")
    state = library_state(library_dir)
    assert state.raw_present is True
    assert state.index_present is True
    assert state.scene_count == 8
