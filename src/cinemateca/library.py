"""cinemateca.library — Film registry + per-film artifact state.

Under the multi-film per-film layout each film lives at::

    <library_dir>/<slug>/
        raw/           — source video file(s)
        metadata/      — keyframes_metadata.json, scene_tags.json, etc.
        embeddings/    — per-film CLIP index

:func:`scan_library` reads the ``films.json`` registry and populates
REAL per-film ``scene_count`` and ``is_processed`` from disk artifacts.
:func:`library_state` reports the aggregate global state (all films).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".m4v", ".webm"}


@dataclass
class Film:
    """One registered film in the library.

    ``scene_count`` reflects the actual number of detected scenes on disk
    (length of ``<slug>/metadata/keyframes_metadata.json``). ``is_processed``
    derives from ``scene_count > 0``. ``year`` is the production year
    stored in ``films.json``, or ``None`` when unknown.
    """

    slug: str
    title: str
    raw_path: Path
    scene_count: int = 0
    is_processed: bool = False
    year: int | None = None


@dataclass
class LibraryState:
    """Aggregate artifact state across all registered films.

    Attributes:
        raw_present: At least one raw video file exists.
        index_present: The CLIP embeddings index file exists.
        scene_count: Total scene count across all films (or the global
            single ``keyframes_metadata.json`` in the legacy flat layout).
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

    The single-film v0.3 honest-limitation (``scene_count == 0`` always,
    ``is_processed is False`` always) is removed here: under the per-film
    layout, ``<library_dir>/<slug>/metadata/keyframes_metadata.json`` is
    the authoritative per-film scene count, and ``is_processed`` derives
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
                    scene_count = len(kf_meta)
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


def library_state(
    raw_dir: Path,
    metadata_dir: Path,
    embeddings_index_path: Path | None = None,
) -> LibraryState:
    """Report the aggregate global artifact state.

    In the legacy flat layout (pre-multi-film migration) this reads the
    single global ``metadata_dir/keyframes_metadata.json``. T4 will
    supersede this with a registry-aware aggregate. Until then the
    signature is unchanged so existing callers keep working.

      * ``raw_present`` — any video file in ``raw_dir``.
      * ``scene_count`` — length of ``metadata_dir/keyframes_metadata.json``
        (0 if absent/empty/malformed).
      * ``index_present`` — ``embeddings_index_path`` exists (``None``
        → reported as absent).
    """
    raw_present = False
    if raw_dir.exists():
        raw_present = any(
            p.suffix.lower() in _VIDEO_EXTENSIONS for p in raw_dir.iterdir()
        )

    scene_count = 0
    kf_path = metadata_dir / "keyframes_metadata.json"
    if kf_path.exists():
        try:
            with open(kf_path, encoding="utf-8") as f:
                kf_meta = json.load(f)
            if isinstance(kf_meta, list):
                scene_count = len(kf_meta)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Unreadable keyframes_metadata.json: %s", exc)

    index_present = bool(
        embeddings_index_path is not None and embeddings_index_path.exists()
    )

    return LibraryState(
        raw_present=raw_present,
        index_present=index_present,
        scene_count=scene_count,
    )


def load_registry(library_dir: Path) -> dict[str, dict]:
    """Load films.json from ``library_dir``; empty dict if absent.

    The registry is the single source of truth for which films exist.
    Per-film derived state (scene_count, is_processed) is NOT stored here;
    it is computed from on-disk artefacts each time :func:`scan_library`
    runs.
    """
    path = library_dir / "films.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        logger.warning("films.json is not a dict; treating as empty: %s", path)
        return {}
    return data


def save_registry(library_dir: Path, registry: dict[str, dict]) -> None:
    """Atomically persist the registry to ``library_dir/films.json``.

    Atomic: write to ``films.json.tmp`` then rename, so a crashed write
    cannot truncate the file.
    """
    library_dir.mkdir(parents=True, exist_ok=True)
    final = library_dir / "films.json"
    tmp = library_dir / "films.json.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False, sort_keys=True)
    tmp.replace(final)


def register_film(
    library_dir: Path,
    *,
    slug: str,
    title: str,
    year: int | None,
    raw_filename: str,
) -> None:
    """Add a film to the registry. Duplicate slug → ``ValueError``."""
    registry = load_registry(library_dir)
    if slug in registry:
        raise ValueError(f"Slug {slug!r} already registered")
    registry[slug] = {
        "slug": slug,
        "title": title,
        "year": year,
        "raw_filename": raw_filename,
        "added_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    save_registry(library_dir, registry)
    logger.info("Registered film: %s (%s)", slug, title)


def delete_film(library_dir: Path, *, slug: str) -> None:
    """Remove a film from the registry. Unknown slug → ``KeyError``.

    Does NOT delete the film's on-disk artefacts — that is the caller's
    decision (idempotent re-add must be possible after a metadata-only
    delete).
    """
    registry = load_registry(library_dir)
    if slug not in registry:
        raise KeyError(f"Slug {slug!r} not in registry")
    del registry[slug]
    save_registry(library_dir, registry)
    logger.info("Deleted film: %s", slug)
