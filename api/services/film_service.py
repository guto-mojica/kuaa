"""Shared film-library helpers used by multiple API services."""
from __future__ import annotations

from pathlib import Path

from cinemateca.library import Film, scan_library


def list_films(library_dir: Path, q: str = "") -> list[Film]:
    """Return registered films, optionally filtered by title/slug substring."""
    films = scan_library(library_dir)
    if q.strip():
        needle = q.strip().lower()
        films = [f for f in films if needle in f.title.lower() or needle in f.slug.lower()]
    return films


def film_by_slug(library_dir: Path, slug: str) -> Film | None:
    """Return the Film for slug, or None if not found."""
    for film in scan_library(library_dir):
        if film.slug == slug:
            return film
    return None
