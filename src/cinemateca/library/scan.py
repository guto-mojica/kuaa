"""Filesystem scan over the library: derive Film + LibraryState from disk."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from cinemateca.library.registry import Film, load_registry

logger = logging.getLogger(__name__)

# Must stay in sync with ``config/default.yaml`` → ``embeddings.filename``.
# Threaded here instead of through ``cfg`` to keep :func:`library_state`'s
# API surface a single argument; the cost is this one cross-file invariant.
_KEYFRAME_EMBEDDINGS_FILENAME = "keyframe_embeddings.npy"


@dataclass
class LibraryState:
    """Aggregate artifact state across all registered films.

    Attributes:
        raw_present: At least one registered film has a raw video on disk.
        index_present: At least one registered film has a per-film embeddings
            index (``<library_dir>/<slug>/embeddings/keyframe_embeddings.npy``).
        scene_count: Total scene count summed across all films.
        is_processed: ``scene_count > 0`` — at least one film has been
            processed.
    """

    raw_present: bool
    index_present: bool
    scene_count: int

    @property
    def is_processed(self) -> bool:
        return self.scene_count > 0


def scan_library(library_dir: Path) -> list[Film]:
    """Return every registered film with REAL per-film disk state.

    Under the per-film layout, ``<library_dir>/<slug>/metadata/keyframes_metadata.json``
    is the authoritative per-film scene count, and ``is_processed`` derives
    from it (``scene_count > 0``).

    Films appear in the result in the order ``films.json`` stores them
    (sorted by slug, because :func:`save_registry` writes ``sort_keys=True``).
    """
    if not library_dir.exists():
        logger.warning("library_dir not found: %s", library_dir)
        return []

    registry = load_registry(library_dir)
    films: list[Film] = []
    for slug, entry in registry.items():
        film_dir = library_dir / slug
        kf_path = film_dir / "metadata" / "keyframes_metadata.json"
        scene_count = 0
        if kf_path.exists():
            try:
                with open(kf_path, encoding="utf-8") as f:
                    kf_meta = json.load(f)
                if isinstance(kf_meta, list):
                    # Count unique scene_id values — the detector emits
                    # N rows per scene (N keyframes for embedding density).
                    scene_count = len({e.get("scene_id") for e in kf_meta})
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "Unreadable keyframes_metadata.json for %s: %s", slug, exc
                )
        films.append(
            Film(
                slug=slug,
                title=entry.get("title", slug),
                raw_path=film_dir / "raw" / entry.get("raw_filename", ""),
                scene_count=scene_count,
                is_processed=scene_count > 0,
                year=entry.get("year"),
            )
        )
        logger.debug("Scanned film: %s (scenes=%d)", slug, scene_count)
    return films


def library_state(library_dir: Path) -> LibraryState:
    """Aggregate state across all films in the registry.

    Fields:

      * ``raw_present`` — at least one film has a raw video on disk.
      * ``index_present`` — at least one film has a per-film embeddings index
        at ``<library_dir>/<slug>/embeddings/keyframe_embeddings.npy``
        (canonical filename from ``config/default.yaml → embeddings.filename``).
      * ``scene_count`` — SUM of scene counts across all films.
    """
    films = scan_library(library_dir)
    if not films:
        return LibraryState(raw_present=False, index_present=False, scene_count=0)
    raw_present = any(f.raw_path.exists() for f in films)
    index_present = any(
        (library_dir / f.slug / "embeddings" / _KEYFRAME_EMBEDDINGS_FILENAME).exists()
        for f in films
    )
    scene_count = sum(f.scene_count for f in films)
    return LibraryState(
        raw_present=raw_present,
        index_present=index_present,
        scene_count=scene_count,
    )
