"""Per-film on-disk context handle.

``FilmContext`` is the single place that answers "where does this film's
artefacts live on disk".  Every route that needs paths goes through this
object; no consumer recomputes ``cfg.paths.*`` inline.

Constructors
------------
Two class-methods build a ``FilmContext``:

* ``for_film(cfg, slug)`` — primary entry point used by the FastAPI layer.
  Reads ``cfg.paths.library_dir`` from the injected config object and
  creates missing subdirectories on first access.  Raises ``ValueError``
  for an unregistered slug (legacy contract — 16 call sites and the
  ``try/except ValueError`` in ``api/deps.py:216`` depend on this).

* ``from_paths(*, library_dir, slug, data_dir)`` — config-free constructor
  added in P3 for service-layer and test code that already holds resolved
  paths.  Raises ``KeyError`` for an unregistered slug, matching the
  ``Library.get_film`` contract.

Exception-contract asymmetry (intentional)
------------------------------------------
``for_film`` raises ``ValueError``; ``from_paths`` raises ``KeyError``.
This asymmetry is preserved for backward compatibility.  A future cleanup
may unify them — see ``docs/superpowers/plans/`` for the P3 plan note.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FilmContext:
    """Resolved on-disk locations for a film's (or the library's) artefacts.

    Attributes:
        slug: Film slug, or ``None`` for the global/flat context (the
            only shape v0.3 builds). A future multi-film epic will
            populate this for per-film resolution.
        raw_path: Directory holding source video files (``cfg.paths.raw_dir``).
            Named ``raw_path`` per the Phase-3a plan spec; it is the raw
            *directory* under the current flat layout, and becomes the
            per-film raw video path once the multi-film epic lands.
        data_dir: Root served at ``/media`` — resolved (``.resolve()``)
            because keyframe-URL math compares resolved paths.
        metadata_dir: Directory of JSON metadata artefacts
            (keyframes/descriptions/tags/visual/annotations).
        frames_dir: Directory of extracted keyframe images.
        embeddings_dir: Directory of the CLIP embeddings index + mapping.
    """

    slug: str | None
    raw_path: Path
    data_dir: Path
    metadata_dir: Path
    frames_dir: Path
    embeddings_dir: Path

    @classmethod
    def from_config(cls, cfg: Any) -> FilmContext:
        """Build the global/flat context from a loaded ``Config``.

        Reproduces, in ONE place, the exact path coercions the routes
        previously did inline:

          * ``data_dir`` is ``.resolve()``-d (keyframe-URL relative-to
            math, and the ``/media`` mount, both used the resolved path).
          * the other dirs are wrapped in ``Path`` but NOT resolved,
            matching the prior route behaviour byte-for-byte (e.g.
            ``Path(cfg.paths.metadata_dir)`` was passed un-resolved to
            ``_load_metadata``).

        This is the only constructor v0.3 provides; ``slug`` is
        ``None`` (global). A future multi-film epic adds the per-film
        variant.
        """
        return cls(
            slug=None,
            raw_path=Path(cfg.paths.raw_dir),
            data_dir=Path(cfg.paths.data_dir).resolve(),
            metadata_dir=Path(cfg.paths.metadata_dir),
            frames_dir=Path(cfg.paths.frames_dir),
            embeddings_dir=Path(cfg.paths.embeddings_dir),
        )

    @classmethod
    def from_paths(
        cls,
        *,
        library_dir: Path,
        slug: str,
        data_dir: Path,
    ) -> FilmContext:
        """Build a per-film context from explicit paths (no cfg object needed).

        ``library_dir`` is the films-registry root; ``data_dir`` is the ``/media``
        mount root (typically one level above ``library_dir``).

        Raises:
            ValueError: slug contains a path-traversal component (``'../x'``).
            KeyError: slug is not in the registry.
        """
        if not slug or slug != Path(slug).name:
            raise ValueError(f"Invalid slug: {slug!r}")
        from cinemateca.library.registry import load_registry

        registry = load_registry(library_dir)
        if slug not in registry:
            raise KeyError(f"Film not registered: {slug!r}")
        film_dir = library_dir / slug
        for sub in ("raw", "metadata", "frames", "embeddings"):
            (film_dir / sub).mkdir(parents=True, exist_ok=True)
        return cls(
            slug=slug,
            raw_path=film_dir / "raw",
            data_dir=Path(data_dir).resolve(),
            metadata_dir=film_dir / "metadata",
            frames_dir=film_dir / "frames",
            embeddings_dir=film_dir / "embeddings",
        )

    @classmethod
    def for_film(cls, cfg: Any, slug: str) -> FilmContext:
        """Build a per-film context from a loaded ``Config`` and a slug.

        The film must exist as a directory under ``cfg.paths.library_dir/``
        — that directory is the boundary of all per-film artefacts.

        Path semantics under the per-film layout:
          * ``data_dir`` is the media-mount root (``cfg.paths.data_dir``),
            ``.resolve()``-d. It must match the directory mounted at
            ``/media`` in :mod:`api.server` so keyframe URLs resolve to
            files the static-files handler actually serves. For real
            metadata that still carries pre-migration absolute paths
            under ``data/frames/...`` this lets URLs resolve to the
            still-present flat files; for relative or new per-film
            absolute paths it resolves to ``/media/library/<slug>/...``.
          * ``raw_path`` / ``metadata_dir`` / ``frames_dir`` / ``embeddings_dir``
            all live under ``<library_dir>/<slug>/...`` and are returned
            un-resolved, matching the flat-context contract byte-for-byte.
        """
        # Reject traversal slugs (e.g. "../secret") before any disk math runs.
        # `Path(slug).name` strips directory components, so a clean slug equals
        # its own .name; "../secret" becomes "secret" and the comparison fails.
        if not slug or slug != Path(slug).name:
            raise ValueError(f"Invalid slug: {slug!r}")
        from cinemateca.library.registry import load_registry

        library_dir = Path(cfg.paths.library_dir)
        film_dir = library_dir / slug
        # Registry is the single gate — unregistered slugs are rejected even
        # if a directory happens to exist on disk (orphaned dirs, manual creates).
        registry = load_registry(library_dir)
        if slug not in registry:
            # Note: raises ValueError for legacy compat (16 callers + api/deps.py:216
            # catch this); from_paths() raises KeyError for new code.
            raise ValueError(f"Film not registered: {slug!r}")
        # Create subdirs on first access for films registered without migration.
        for sub in ("raw", "metadata", "frames", "embeddings"):
            (film_dir / sub).mkdir(parents=True, exist_ok=True)
        # data_dir defaults to library_dir for test configs that don't supply
        # cfg.paths.data_dir; real configs always carry data_dir (default.yaml).
        data_dir_str = getattr(cfg.paths, "data_dir", None) or str(library_dir)
        return cls(
            slug=slug,
            raw_path=film_dir / "raw",
            data_dir=Path(data_dir_str).resolve(),
            metadata_dir=film_dir / "metadata",
            frames_dir=film_dir / "frames",
            embeddings_dir=film_dir / "embeddings",
        )
