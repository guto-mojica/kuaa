"""Unit tests for cinemateca.library.Library — the typed handle."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cinemateca.library import (
    Film,
    FilmContext,
    Library,
    LibraryState,
    register_film,
)


def test_library_construction(tmp_path):
    lib = Library(library_dir=tmp_path)
    assert lib.library_dir == tmp_path


def test_list_films_empty(tmp_path):
    lib = Library(library_dir=tmp_path)
    assert lib.list_films() == []


def test_list_films_after_register(tmp_path):
    lib = Library(library_dir=tmp_path)
    lib.register(slug="alpha", title="Alpha", year=2026, raw_filename="alpha.mp4")
    films = lib.list_films()
    assert len(films) == 1
    assert films[0].slug == "alpha"


def test_get_film_returns_film(tmp_path):
    lib = Library(library_dir=tmp_path)
    lib.register(slug="alpha", title="Alpha", year=2026, raw_filename="alpha.mp4")
    f = lib.get_film("alpha")
    assert isinstance(f, Film)
    assert f.title == "Alpha"


def test_get_film_unknown_raises(tmp_path):
    lib = Library(library_dir=tmp_path)
    with pytest.raises(KeyError, match="alpha"):
        lib.get_film("alpha")


def test_remove_film(tmp_path):
    lib = Library(library_dir=tmp_path)
    lib.register(slug="alpha", title="Alpha", year=2026, raw_filename="alpha.mp4")
    lib.remove("alpha")
    assert lib.list_films() == []


def test_remove_unknown_raises(tmp_path):
    lib = Library(library_dir=tmp_path)
    with pytest.raises(KeyError, match="alpha"):
        lib.remove("alpha")


def test_state_empty_returns_zero(tmp_path):
    lib = Library(library_dir=tmp_path)
    assert lib.state() == LibraryState(
        raw_present=False, index_present=False, scene_count=0
    )


def test_context_for_registered_film(tmp_path):
    lib = Library(library_dir=tmp_path)
    lib.register(slug="alpha", title="Alpha", year=2026, raw_filename="alpha.mp4")
    # Library.context() needs a data_dir for the FilmContext data_dir field.
    ctx = lib.context("alpha", data_dir=tmp_path)
    assert isinstance(ctx, FilmContext)
    assert ctx.slug == "alpha"
    assert ctx.metadata_dir == tmp_path / "alpha" / "metadata"


def test_context_unregistered_raises(tmp_path):
    lib = Library(library_dir=tmp_path)
    with pytest.raises(ValueError, match="Film not registered"):
        lib.context("ghost", data_dir=tmp_path)
