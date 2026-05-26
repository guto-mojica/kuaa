"""cinemateca.library — film registry, scan, per-film context, on-disk utilities.

Public API:

    from cinemateca.library import (
        Library,                        # typed handle (T10)
        Film, LibraryState, FilmContext,
        scan_library, library_state,
        register_film, delete_film, load_registry, save_registry,
        load_json, keyframe_url, to_smpte, derive_fps,
        load_tag_index, load_metadata,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cinemateca.library.context import FilmContext
from cinemateca.library.metadata import (
    load_metadata,
    load_tag_index,
)
from cinemateca.library.paths import (
    derive_fps,
    keyframe_url,
    load_json,
    to_smpte,
)
from cinemateca.library.registry import (
    Film,
    delete_film,
    load_registry,
    register_film,
    save_registry,
)
from cinemateca.library.scan import (
    LibraryState,
    library_state,
    scan_library,
)


@dataclass(frozen=True)
class Library:
    """Typed handle for the film registry + scan operations.

    Wraps a ``library_dir`` so call sites don't have to thread the
    Path through every function. Methods delegate to the module-level
    functions; this is an additive thin wrapper, not a replacement.
    """

    library_dir: Path

    def list_films(self) -> list[Film]:
        return scan_library(self.library_dir)

    def get_film(self, slug: str) -> Film:
        for f in self.list_films():
            if f.slug == slug:
                return f
        raise KeyError(f"Film not registered: {slug!r}")

    def context(self, slug: str, *, data_dir: Path) -> FilmContext:
        """Build a FilmContext for ``slug``.

        ``data_dir`` is required — it is the ``/media`` mount root for
        keyframe URL resolution. In production this is ``cfg.paths.data_dir``,
        typically ONE LEVEL ABOVE ``library_dir``.

        Raises:
            ValueError: slug is invalid (traversal).
            KeyError: slug is not registered.
        """
        return FilmContext.from_paths(
            library_dir=self.library_dir,
            slug=slug,
            data_dir=data_dir,
        )

    def register(
        self,
        *,
        slug: str,
        title: str,
        year: int | None,
        raw_filename: str,
    ) -> None:
        register_film(
            self.library_dir,
            slug=slug,
            title=title,
            year=year,
            raw_filename=raw_filename,
        )

    def remove(self, slug: str) -> None:
        delete_film(self.library_dir, slug=slug)

    def state(self) -> LibraryState:
        return library_state(self.library_dir)


__all__ = [
    "Film",
    "FilmContext",
    "Library",
    "LibraryState",
    "delete_film",
    "derive_fps",
    "keyframe_url",
    "library_state",
    "load_json",
    "load_metadata",
    "load_registry",
    "load_tag_index",
    "register_film",
    "save_registry",
    "scan_library",
    "to_smpte",
]
