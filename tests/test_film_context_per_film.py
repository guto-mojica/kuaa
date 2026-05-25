"""Per-film FilmContext path resolution."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from cinemateca.library import FilmContext


def _cfg(library_dir: Path, data_dir: Path | None = None) -> object:
    """Test config namespace.

    ``data_dir`` defaults to ``library_dir`` (the legacy fallback inside
    ``FilmContext.for_film`` for minimal test configs); pass a separate
    path to exercise the realistic shape where the media-mount root and
    the per-film library root differ.
    """
    paths_kw = {"library_dir": str(library_dir)}
    if data_dir is not None:
        paths_kw["data_dir"] = str(data_dir)
    return SimpleNamespace(paths=SimpleNamespace(**paths_kw))


def test_for_film_resolves_per_film_paths(tmp_path: Path) -> None:
    """Per-film artefact dirs land under ``<library_dir>/<slug>/...`` and
    ``data_dir`` is the media-mount root (``cfg.paths.data_dir``), NOT
    the library root — keyframe URLs must resolve against the directory
    mounted at ``/media`` in :mod:`api.server`."""
    from cinemateca.library import register_film

    data_dir = tmp_path / "data"
    library_dir = data_dir / "library"
    library_dir.mkdir(parents=True)
    register_film(library_dir, slug="jeca_tatu", title="Jeca Tatu", year=1959, raw_filename="jeca_tatu.mp4")

    ctx = FilmContext.for_film(_cfg(library_dir, data_dir=data_dir), "jeca_tatu")

    assert ctx.slug == "jeca_tatu"
    assert ctx.raw_path == library_dir / "jeca_tatu" / "raw"
    assert ctx.metadata_dir == library_dir / "jeca_tatu" / "metadata"
    assert ctx.frames_dir == library_dir / "jeca_tatu" / "frames"
    assert ctx.embeddings_dir == library_dir / "jeca_tatu" / "embeddings"
    assert ctx.data_dir == data_dir.resolve()


def test_for_film_unknown_slug_raises(tmp_path: Path) -> None:
    library_dir = tmp_path / "library"
    library_dir.mkdir()
    with pytest.raises(ValueError, match="Film not registered"):
        FilmContext.for_film(_cfg(library_dir), "ghost")


def test_for_film_rejects_traversal_slug(tmp_path: Path) -> None:
    """Slugs containing path separators or dot components are rejected
    before any disk math runs — closes a traversal attack vector before
    T9 wires this to user-controlled HTTP input."""
    library_dir = tmp_path / "library"
    library_dir.mkdir()
    for bad_slug in ("../secret", "a/b", "", "."):
        with pytest.raises(ValueError, match="Invalid slug"):
            FilmContext.for_film(_cfg(library_dir), bad_slug)
