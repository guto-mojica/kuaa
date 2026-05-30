"""FilmContext audio-path properties (audio_dir / segments)."""

from __future__ import annotations

from pathlib import Path


def _make_ctx(tmp_path: Path):
    """Build a FilmContext against the real frozen-dataclass signature.

    Fields: slug, raw_path, data_dir, metadata_dir, frames_dir, embeddings_dir.
    The audio properties derive from ``metadata_dir.parent`` (the film root),
    so metadata_dir is placed under a ``demo`` film dir.
    """
    from cinemateca.library.context import FilmContext

    film_dir = tmp_path / "demo"
    return FilmContext(
        slug="demo",
        raw_path=film_dir / "raw",
        data_dir=tmp_path,
        metadata_dir=film_dir / "metadata",
        frames_dir=film_dir / "frames",
        embeddings_dir=film_dir / "embeddings",
    )


def test_audio_dir_property_present():
    from cinemateca.library.context import FilmContext

    assert hasattr(FilmContext, "audio_dir")
    assert hasattr(FilmContext, "audio_segments_dir")


def test_audio_dir_under_film_root(tmp_path):
    ctx = _make_ctx(tmp_path)
    assert ctx.audio_dir == ctx.metadata_dir.parent / "audio"
    assert ctx.audio_segments_dir == ctx.audio_dir / "segments"
