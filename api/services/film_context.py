"""Resolved artifact-path context for a film (or the global library).

``FilmContext`` is the single place that answers "where does this film's
data live on disk". Before Phase 3a every route recomputed
``Path(cfg.paths.metadata_dir)`` / ``Path(cfg.paths.data_dir).resolve()``
inline; that scattered path math is now centralized here.

Multi-film extension point (post-recovery epic — NOT v0.3)
----------------------------------------------------------
The data layout is FLAT and single-film/global: every film shares one
``metadata_dir`` / ``frames_dir`` / ``embeddings_dir`` (see
``cinemateca.library``'s single-film note). The Phase-5 maintainer
decision is that **v0.3 stays SINGLE-FILM** with an honest library
placeholder; a true per-film data model is deferred to a later
multi-film epic. This class is still modelled so that future move is
additive, NOT a rewrite of call sites — keep this seam:

  * ``slug`` is already carried (currently ``None`` for the global
    context) so a future ``FilmContext.for_film(cfg, slug)`` can resolve
    per-film subdirectories while keeping the same attribute surface.
  * Path resolution goes through this object, so when the multi-film
    epic changes where a film's artefacts live, only :meth:`from_config`
    / a new ``for_film`` constructor changes — every consumer
    (catalog/annotations/search services) is untouched.

Phase 3a deliberately implements ONLY the global/flat constructor. It
does NOT implement multi-film resolution — that is a future product
decision.
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
