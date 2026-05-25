"""Film registry — films.json read/write + Film dataclass."""
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
