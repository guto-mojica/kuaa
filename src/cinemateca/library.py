"""cinemateca.library — Film inventory + per-film artifact state.

Multi-film layout (v0.3+): each processed film lives under
``data/films/{slug}/`` with its own ``metadata/``, ``frames/``,
and ``embeddings/`` subdirectories.  :func:`scan_library` merges:

  1. Registered films — subdirectories in ``films_dir`` (``data/films/``).
     Each may have an optional ``film.json`` carrying a custom title and
     raw-video path; otherwise the title is derived from the slug and the
     raw path is located by stem-match in ``raw_dir``.
  2. Unregistered raw videos — video files in ``raw_dir`` whose stem does
     not already appear in ``films_dir``.

Per-film scene counts are read from
``films_dir/{slug}/metadata/keyframes_metadata.json`` and are therefore
honest, not fabricated.  :func:`library_state` still reports the ONE
global artifact set (used as fallback when no film is selected).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".m4v", ".webm"}


@dataclass
class Film:
    """One film in the library.

    ``scene_count`` and ``is_processed`` reflect per-film state when the
    film has a directory under ``films_dir``; they remain ``0`` /
    ``False`` for unregistered raw-only entries.
    """

    slug: str
    title: str
    raw_path: Path
    scene_count: int = 0
    is_processed: bool = False
    is_registered: bool = False  # True when film.json exists in films_dir


@dataclass
class LibraryState:
    """Honest GLOBAL artifact state (used when no film is selected).

    Attributes:
        raw_present: At least one raw video file exists in ``raw_dir``.
        index_present: The CLIP embeddings index file exists.
        scene_count: Number of scenes in the global flat
            ``keyframes_metadata.json`` (0 if absent/empty).
        is_processed: ``scene_count > 0``.
    """

    raw_present: bool
    index_present: bool
    scene_count: int

    @property
    def is_processed(self) -> bool:
        return self.scene_count > 0


def _read_scene_count(metadata_dir: Path) -> int:
    """Return the number of scenes in ``metadata_dir/keyframes_metadata.json``."""
    kf_path = metadata_dir / "keyframes_metadata.json"
    if not kf_path.exists():
        return 0
    try:
        with open(kf_path, encoding="utf-8") as f:
            data = json.load(f)
        return len(data) if isinstance(data, list) else 0
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Unreadable keyframes_metadata.json: %s", exc)
        return 0


def _find_raw_video(raw_dir: Path, slug: str) -> Path:
    """Return the raw video file whose stem slug matches ``slug``.

    Handles names with spaces or mixed case (e.g. ``Soluçonfonia.mp4``
    whose slug is ``soluçonfonia``).  Falls back to a synthesised path
    (``raw_dir / slug``) when no file is found so callers always get a
    ``Path`` object.
    """
    if raw_dir.exists():
        for f in raw_dir.iterdir():
            if f.suffix.lower() in _VIDEO_EXTENSIONS:
                if f.stem.lower().replace(" ", "_") == slug:
                    return f
    return raw_dir / slug


def scan_library(
    raw_dir: Path,
    metadata_dir: Path,
    films_dir: Path | None = None,
) -> list[Film]:
    """Return the film inventory.

    When ``films_dir`` is provided (and exists), registered per-film
    directories are enumerated first (with real scene counts); then any
    raw video whose slug is not already represented is appended as an
    unregistered entry (scene_count=0).

    When ``films_dir`` is omitted or absent, only ``raw_dir`` is scanned
    — identical to the old single-film behaviour.
    """
    films: list[Film] = []
    registered: set[str] = set()

    # ── Registered per-film dirs ──────────────────────────────────────
    if films_dir and films_dir.exists():
        for film_dir in sorted(films_dir.iterdir()):
            if not film_dir.is_dir() or film_dir.name.startswith("."):
                continue
            slug = film_dir.name

            # film.json is required — orphan pipeline dirs (no registration)
            # are skipped so they don't pollute the library list.
            film_json = film_dir / "film.json"
            if not film_json.exists():
                continue
            title: str | None = None
            raw_path: Path | None = None
            try:
                meta = json.loads(film_json.read_text(encoding="utf-8"))
                title = meta.get("title")
                rp = meta.get("raw_path")
                raw_path = Path(rp) if rp else None
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Unreadable film.json for %s: %s", slug, exc)

            if title is None:
                title = slug.replace("_", " ").title()
            if raw_path is None:
                raw_path = _find_raw_video(raw_dir, slug)

            scene_count = _read_scene_count(film_dir / "metadata")
            films.append(Film(
                slug=slug,
                title=title,
                raw_path=raw_path,
                scene_count=scene_count,
                is_processed=scene_count > 0,
                is_registered=True,
            ))
            registered.add(slug)
            logger.debug("Registered film: %s (%d scenes)", slug, scene_count)

    # ── Unregistered raw videos ───────────────────────────────────────
    if raw_dir.exists():
        for video_path in sorted(raw_dir.iterdir()):
            if video_path.suffix.lower() not in _VIDEO_EXTENSIONS:
                continue
            slug = video_path.stem.lower().replace(" ", "_")
            if slug in registered:
                continue
            films.append(Film(
                slug=slug,
                title=video_path.stem.replace("_", " ").title(),
                raw_path=video_path,
            ))
            logger.debug("Unregistered film: %s", slug)

    return films


def library_state(
    raw_dir: Path,
    metadata_dir: Path,
    embeddings_index_path: Path | None = None,
) -> LibraryState:
    """Report the honest GLOBAL artifact state.

    Used as a fallback when no film is selected (global flat layout).
    """
    raw_present = False
    if raw_dir.exists():
        raw_present = any(
            p.suffix.lower() in _VIDEO_EXTENSIONS for p in raw_dir.iterdir()
        )

    scene_count = _read_scene_count(metadata_dir)

    index_present = bool(
        embeddings_index_path is not None and embeddings_index_path.exists()
    )

    return LibraryState(
        raw_present=raw_present,
        index_present=index_present,
        scene_count=scene_count,
    )
