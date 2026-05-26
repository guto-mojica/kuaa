"""Tests for scripts/migrate_flat_to_library.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.migrate_flat_to_library import migrate_flat_to_library


def _populate_flat_layout(root: Path) -> None:
    """Build a minimal flat data/ layout with one video + minimal artefacts."""
    (root / "raw").mkdir(parents=True)
    (root / "raw" / "jeca_tatu.mp4").write_bytes(b"video-bytes")
    (root / "metadata").mkdir()
    (root / "metadata" / "keyframes_metadata.json").write_text(
        json.dumps([{"scene_id": 0, "filepath": "data/frames/keyframes/0.jpg"}]),
        encoding="utf-8",
    )
    (root / "frames" / "keyframes").mkdir(parents=True)
    (root / "frames" / "keyframes" / "0.jpg").write_bytes(b"img-bytes")
    (root / "embeddings").mkdir()
    (root / "embeddings" / "keyframe_embeddings.npy").write_bytes(b"npy-bytes")
    (root / "embeddings" / "index_mapping.json").write_text(
        json.dumps([{"keyframe_path": "data/frames/keyframes/0.jpg"}]),
        encoding="utf-8",
    )


def test_migrate_moves_artefacts_into_per_film_layout(tmp_path: Path) -> None:
    flat = tmp_path / "data"
    _populate_flat_layout(flat)
    library = tmp_path / "data" / "library"

    migrate_flat_to_library(
        flat_root=flat,
        library_dir=library,
        slug="jeca_tatu",
        title="Jeca Tatu",
        year=1959,
    )

    assert (library / "films.json").exists()
    registry = json.loads((library / "films.json").read_text())
    assert "jeca_tatu" in registry
    assert (library / "jeca_tatu" / "raw" / "jeca_tatu.mp4").read_bytes() == b"video-bytes"
    assert (library / "jeca_tatu" / "metadata" / "keyframes_metadata.json").exists()
    assert (library / "jeca_tatu" / "frames" / "keyframes" / "0.jpg").exists()
    assert (library / "jeca_tatu" / "embeddings" / "keyframe_embeddings.npy").exists()


def test_migrate_is_idempotent(tmp_path: Path) -> None:
    """Running the migration twice does not duplicate or error."""
    flat = tmp_path / "data"
    _populate_flat_layout(flat)
    library = tmp_path / "data" / "library"
    migrate_flat_to_library(
        flat_root=flat, library_dir=library, slug="jeca_tatu", title="J", year=1959
    )
    # Second call must not raise:
    migrate_flat_to_library(
        flat_root=flat, library_dir=library, slug="jeca_tatu", title="J", year=1959
    )
    assert (library / "jeca_tatu" / "raw" / "jeca_tatu.mp4").exists()


def test_migrate_rejects_when_no_raw_video(tmp_path: Path) -> None:
    flat = tmp_path / "data"
    (flat / "raw").mkdir(parents=True)
    library = tmp_path / "data" / "library"
    with pytest.raises(FileNotFoundError, match="No raw video"):
        migrate_flat_to_library(
            flat_root=flat,
            library_dir=library,
            slug="ghost",
            title="Ghost",
            year=None,
        )
