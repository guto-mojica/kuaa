"""Per-film FilmContext path resolution."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from api.services.film_context import FilmContext


def _cfg(library_dir: Path) -> object:
    return SimpleNamespace(paths=SimpleNamespace(library_dir=str(library_dir)))


def test_for_film_resolves_per_film_paths(tmp_path: Path) -> None:
    library_dir = tmp_path / "library"
    (library_dir / "jeca_tatu" / "raw").mkdir(parents=True)
    (library_dir / "jeca_tatu" / "raw" / "jeca_tatu.mp4").write_bytes(b"")
    (library_dir / "jeca_tatu" / "metadata").mkdir()
    (library_dir / "jeca_tatu" / "frames").mkdir()
    (library_dir / "jeca_tatu" / "embeddings").mkdir()

    ctx = FilmContext.for_film(_cfg(library_dir), "jeca_tatu")

    assert ctx.slug == "jeca_tatu"
    assert ctx.raw_path == library_dir / "jeca_tatu" / "raw"
    assert ctx.metadata_dir == library_dir / "jeca_tatu" / "metadata"
    assert ctx.frames_dir == library_dir / "jeca_tatu" / "frames"
    assert ctx.embeddings_dir == library_dir / "jeca_tatu" / "embeddings"
    # data_dir is the LIBRARY root (used by keyframe_url for /media/<slug>/...)
    assert ctx.data_dir == library_dir.resolve()


def test_for_film_unknown_slug_raises(tmp_path: Path) -> None:
    library_dir = tmp_path / "library"
    library_dir.mkdir()
    with pytest.raises(ValueError, match="No such film directory"):
        FilmContext.for_film(_cfg(library_dir), "ghost")
