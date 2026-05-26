"""Unit tests for cinemateca.library.Library — the typed handle."""

from __future__ import annotations

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
    assert lib.state() == LibraryState(raw_present=False, index_present=False, scene_count=0)


def test_context_for_registered_film(tmp_path):
    lib = Library(library_dir=tmp_path)
    lib.register(slug="alpha", title="Alpha", year=2026, raw_filename="alpha.mp4")
    # Library.context() needs a data_dir for the FilmContext data_dir field.
    ctx = lib.context("alpha", data_dir=tmp_path)
    assert isinstance(ctx, FilmContext)
    assert ctx.slug == "alpha"
    assert ctx.metadata_dir == tmp_path / "alpha" / "metadata"


def test_context_unregistered_raises_keyerror(tmp_path):
    """Library.context now raises KeyError (consistent with get_film)."""
    lib = Library(library_dir=tmp_path)
    with pytest.raises(KeyError, match="Film not registered"):
        lib.context("ghost", data_dir=tmp_path)


def test_filmcontext_from_paths_constructor(tmp_path):
    """FilmContext.from_paths builds a context without a cfg shim."""
    from cinemateca.library.context import FilmContext

    register_film(tmp_path, slug="alpha", title="Alpha", year=2026, raw_filename="alpha.mp4")
    ctx = FilmContext.from_paths(library_dir=tmp_path, slug="alpha", data_dir=tmp_path)
    assert ctx.slug == "alpha"
    assert ctx.metadata_dir == tmp_path / "alpha" / "metadata"
    assert ctx.data_dir == tmp_path.resolve()


def test_filmcontext_from_paths_unregistered_raises_keyerror(tmp_path):
    """from_paths raises KeyError (not ValueError) for unregistered slug."""
    with pytest.raises(KeyError, match="ghost"):
        FilmContext.from_paths(library_dir=tmp_path, slug="ghost", data_dir=tmp_path)


def test_filmcontext_from_paths_invalid_slug_raises_valueerror(tmp_path):
    """from_paths raises ValueError for traversal slugs (not KeyError)."""
    with pytest.raises(ValueError, match="Invalid slug"):
        FilmContext.from_paths(library_dir=tmp_path, slug="../etc", data_dir=tmp_path)
