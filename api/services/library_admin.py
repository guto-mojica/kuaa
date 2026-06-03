"""Library admin orchestration extracted from ``api/routes/library.py`` (A2 / Task 5).

Handles the non-HTTP half of film registration (slug derivation, symlink
creation, registry write) and film removal. The route retains all
``Form``/``Response``/``HX-Redirect`` shaping.

Raises ``ValueError`` on invalid input; ``HTTPException(400)`` when the video
path constraint is violated. A4 (Task 8) will map these to the
``CinematecaError`` envelope once F2 is available.
"""

from __future__ import annotations

from pathlib import Path


def resolve_video_path(video_path_str: str, raw_dir_str: str) -> Path:
    """Resolve a video path string to an absolute Path.

    Accepts a bare filename and resolves it against ``raw_dir_str`` when the
    path is not absolute. Returns the expanded + resolved Path. Does NOT check
    existence — callers validate afterwards.
    """
    video = Path(video_path_str.strip()).expanduser()
    if not video.is_absolute():
        candidate = Path(raw_dir_str) / video
        if candidate.exists():
            return candidate
    return video


def register_and_symlink(
    library_dir: Path,
    video: Path,
    slug: str,
    film_title: str,
    raw_dir: Path | None = None,
) -> None:
    """Register the film in the registry and create the per-film raw symlink.

    The symlink ``library/<slug>/raw/<filename>`` points to the resolved
    absolute path of *video*, which may live anywhere on disk — not just in
    the configured raw directory.

    Raises ``ValueError`` when the slug is already registered.
    """
    from cinemateca.library import register_film

    register_film(
        library_dir,
        slug=slug,
        title=film_title,
        year=None,
        raw_filename=video.name,
    )

    per_film_raw = library_dir / slug / "raw"
    per_film_raw.mkdir(parents=True, exist_ok=True)
    link = per_film_raw / video.name
    if not link.exists() and not link.is_symlink():
        link.symlink_to(video.resolve())


def remove_film_and_wipe(library_dir: Path, slug: str, *, wipe: bool) -> None:
    """Deregister a film from the registry and optionally delete its data dir.

    No-ops gracefully when the slug is not found in the registry.
    """
    import shutil

    from cinemateca.library import delete_film, load_registry

    registry = load_registry(library_dir)
    if slug in registry:
        delete_film(library_dir, slug=slug)

    if wipe:
        film_dir = library_dir / slug
        if film_dir.exists():
            shutil.rmtree(film_dir)
