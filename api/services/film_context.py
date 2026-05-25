"""Back-compat re-export shim — FilmContext relocated to cinemateca.library.context.

Old import path ``from api.services.film_context import FilmContext`` keeps
working through this shim. Each importer migrates to
``from cinemateca.library import FilmContext`` lazily; the shim deletes
once all importers have migrated (P2 task T9).
"""
from cinemateca.library.context import FilmContext  # noqa: F401

__all__ = ["FilmContext"]
