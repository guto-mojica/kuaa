"""cinemateca.library — Film inventory + honest global artifact state.

v0.3 is SINGLE-FILM / FLAT: there is exactly ONE global artifact set
(``metadata_dir/keyframes_metadata.json``, the embeddings index, etc.)
shared by whatever video sits in ``raw_dir``. There is no per-film data
model yet (that is the post-recovery multi-film epic; see
``api.services.film_context.FilmContext`` for the extension seam).

Accordingly this module does NOT pretend each raw video has independent
scene counts or its own processed state:

  * :func:`scan_library` returns the raw videos as a plain INVENTORY.
    Every ``Film`` has ``scene_count == 0`` and ``is_processed is False``
    — those per-film fields are kept on the dataclass only for a stable
    signature (callers/tests depend on the shape) but are deliberately
    never populated, because per-film state does not exist in v0.3.
  * :func:`library_state` reports the REAL, GLOBAL artifact state once:
    is a raw video present, is the search index present, and the single
    global scene count from ``keyframes_metadata.json``.
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
    """One raw video file in the inventory.

    ``scene_count`` / ``is_processed`` exist for signature stability only
    and are ALWAYS ``0`` / ``False`` in v0.3 — there is no per-film
    processed state under the flat single-film layout. Read global
    processing state from :func:`library_state` instead.
    """

    slug: str
    title: str
    raw_path: Path
    scene_count: int = 0
    is_processed: bool = False


@dataclass
class LibraryState:
    """Honest GLOBAL artifact state for the single-film v0.3 layout.

    Attributes:
        raw_present: At least one raw video file exists in ``raw_dir``.
        index_present: The CLIP embeddings index file exists.
        scene_count: Number of scenes in the single global
            ``keyframes_metadata.json`` (0 if absent/empty).
        is_processed: ``scene_count > 0`` — the library has a processed
            artifact set. Global, NOT per-film.
    """

    raw_present: bool
    index_present: bool
    scene_count: int

    @property
    def is_processed(self) -> bool:
        return self.scene_count > 0


def scan_library(raw_dir: Path, metadata_dir: Path) -> list[Film]:
    """Return the raw videos in ``raw_dir`` as a plain inventory.

    ``metadata_dir`` is accepted for signature stability (callers pass it)
    but is intentionally NOT consulted: per-film scene counts / processed
    flags do not exist in the flat single-film v0.3 layout, so fabricating
    them per video would be dishonest. Every returned ``Film`` therefore
    has ``scene_count == 0`` and ``is_processed is False``. Use
    :func:`library_state` for the real global artifact state.
    """
    if not raw_dir.exists():
        logger.warning("raw_dir not found: %s", raw_dir)
        return []

    films: list[Film] = []
    for video_path in sorted(raw_dir.iterdir()):
        if video_path.suffix.lower() not in _VIDEO_EXTENSIONS:
            continue
        slug = video_path.stem.lower().replace(" ", "_")
        films.append(
            Film(
                slug=slug,
                title=video_path.stem.replace("_", " ").title(),
                raw_path=video_path,
            )
        )
        logger.debug("Inventory film: %s", slug)

    return films


def library_state(
    raw_dir: Path,
    metadata_dir: Path,
    embeddings_index_path: Path | None = None,
) -> LibraryState:
    """Report the honest GLOBAL artifact state (single-film v0.3 layout).

    There is exactly one of each artifact, shared globally:

      * ``raw_present`` — any video file in ``raw_dir``.
      * ``scene_count`` — length of the single global
        ``metadata_dir/keyframes_metadata.json`` (0 if absent/empty/
        malformed).
      * ``index_present`` — ``embeddings_index_path`` exists (when the
        caller supplies it; ``None`` → reported as absent, callers that
        do not know the embeddings filename simply omit it).
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
    Per-film derived state (scene_count, processed?) is NOT stored here;
    it is read from on-disk artefacts each time via :func:`scan_library`.
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
