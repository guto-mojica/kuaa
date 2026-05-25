"""Search service — thin HTTP adapter over :mod:`cinemateca.search`.

After P1's deep-modules refactor (T3–T13) the search domain logic lives
in :mod:`cinemateca.search`. This module is the HTTP-adapter surface the
route layer (``api/routes/search.py``) and the legacy test suite import
through. It re-exports the symbols the FastAPI app and tests pin against,
and owns three small wrappers that need the FastAPI app config:

  * :func:`_get_embedder` — module-scope so unit tests monkeypatch it to
    bypass real CLIP-model load;
  * :func:`_get_search_index` — resolves per-film embeddings dir +
    ``cfg.embeddings`` filenames, then delegates to :func:`load_index`;
  * :func:`_get_bm25_index_for_ctx` — resolves ``cfg.search.bm25``
    tunables, then delegates to
    :func:`cinemateca.search.bm25.bm25_index_for_ctx`;
  * :func:`has_indexed_films` — library-wide "any OK index?" probe used
    by the route to distinguish "no index yet" from "no hits".
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from api.services.catalog import keyframe_url  # noqa: F401  — used by routes
from api.services.film_context import FilmContext

# Result conversion + Mojica context + films-by-id lookup (T8).
from cinemateca.search._lookup import (
    build_search_context,  # noqa: F401
    build_search_context_aggregate,  # noqa: F401
    films_by_id_lookup,  # noqa: F401
)
from cinemateca.search._results import results_to_dicts  # noqa: F401

# Aggregate cross-film search (T11) — still reads ``_get_embedder`` and
# ``_get_search_index`` off this module via lazy attribute access, so the
# monkeypatches on ``api.services.search._get_*`` keep hitting the call path.
from cinemateca.search.aggregate import aggregate_search  # noqa: F401

# BM25 loader + lru_cache (T7) — module self-registers its cache flusher
# with cinemateca.search.cache so ``clear_index_cache()`` flushes BM25.
from cinemateca.search.bm25 import (
    _cached_bm25_index,  # noqa: F401  — legacy name for tests
    _file_stamp,  # noqa: F401  — legacy name for tests
)

# CLIP search-index loader + mtime/size cache (T6).
from cinemateca.search.cache import (
    IndexStatus,  # noqa: F401
    SearchIndex,  # noqa: F401
    clear_index_cache,  # noqa: F401  — flushes CLIP + BM25
    load_index,  # noqa: F401
)

# CLIP search verbs (T9).
from cinemateca.search.clip import (
    search_image,  # noqa: F401
    search_text,  # noqa: F401
)

# Degenerate-tag display filter (T4).
from cinemateca.search.display import (
    filter_degenerate_tags as _filter_degenerate_tags,  # noqa: F401
)

# Hybrid dispatch (T10).
from cinemateca.search.hybrid import search_hybrid  # noqa: F401

# Upload validation (T5). UploadRejected re-exported for the legacy
# ``api.services.search.UploadRejected`` import path used by routes + tests.
from cinemateca.search.types import UploadRejected  # noqa: F401
from cinemateca.search.upload import (
    MAX_UPLOAD_BYTES,  # noqa: F401
    validate_upload,  # noqa: F401
)

if TYPE_CHECKING:
    from cinemateca.retrieval.bm25 import BM25Index

logger = logging.getLogger(__name__)

# Canonical filenames for the per-film CLIP index. Mirror
# ``config/default.yaml`` → ``embeddings.*``; used as defaults when
# ``cfg.embeddings`` is absent (unit tests with minimal configs).
_DEFAULT_EMBEDDINGS_FILENAME = "keyframe_embeddings.npy"
_DEFAULT_MAPPING_FILENAME = "index_mapping.json"


def _get_bm25_index_for_ctx(ctx: FilmContext) -> BM25Index:
    """Load + cache the BM25 index for one film, using app-config tunables.

    Resolves ``cfg.search.bm25`` for ``stopwords_lang`` / ``k1`` / ``b``
    via :func:`api.deps.get_config`, then forwards to
    :func:`cinemateca.search.bm25.bm25_index_for_ctx`. ``get_config`` is
    imported lazily so this module stays loadable without the FastAPI
    app config wired up (matters for unit tests).
    """
    from api.deps import get_config
    from cinemateca.search.bm25 import bm25_index_for_ctx

    cfg = get_config()
    bm25_cfg = getattr(cfg.search, "bm25", None)
    stopwords_lang = getattr(bm25_cfg, "stopwords_lang", None) if bm25_cfg else None
    k1 = float(getattr(bm25_cfg, "k1", 1.5)) if bm25_cfg else 1.5
    b = float(getattr(bm25_cfg, "b", 0.75)) if bm25_cfg else 0.75
    return bm25_index_for_ctx(ctx, stopwords_lang=stopwords_lang, k1=k1, b=b)


def _get_embedder(cfg: Any) -> object:
    """Return a fresh ``OpenClipEmbedder`` instance.

    Module-scope so unit tests monkeypatch
    ``api.services.search._get_embedder`` to avoid loading the real CLIP
    model. ``cfg`` is accepted for API consistency / future routing but
    currently ignored.
    """
    from cinemateca.models.clip.openclip import OpenClipEmbedder

    return OpenClipEmbedder()


def _get_search_index(cfg: Any, slug: str) -> SearchIndex:
    """Return the (cached) :class:`SearchIndex` for the film identified by *slug*.

    Resolves the per-film embeddings dir via :meth:`FilmContext.for_film`,
    then delegates to :func:`load_index` with canonical filenames
    (``cfg.embeddings.*`` if present, otherwise the module-level
    defaults for minimal test configs).
    """
    emb_cfg = getattr(cfg, "embeddings", None)
    embeddings_filename = (
        getattr(emb_cfg, "filename", _DEFAULT_EMBEDDINGS_FILENAME)
        if emb_cfg is not None
        else _DEFAULT_EMBEDDINGS_FILENAME
    )
    mapping_filename = (
        getattr(emb_cfg, "mapping_filename", _DEFAULT_MAPPING_FILENAME)
        if emb_cfg is not None
        else _DEFAULT_MAPPING_FILENAME
    )
    ctx = FilmContext.for_film(cfg, slug)
    return load_index(
        ctx,
        embeddings_filename=embeddings_filename,
        mapping_filename=mapping_filename,
    )


def has_indexed_films(cfg: Any) -> bool:
    """``True`` iff at least one registered film has an OK :class:`SearchIndex`.

    Lets the route distinguish two empty-hit cases:

      * no indexed films yet → render "No search index found" (user must
        run the embeddings pipeline);
      * indexed films exist but the query produced zero hits above
        ``min_similarity`` → render "No results".
    """
    from cinemateca.library import scan_library

    library_dir = Path(cfg.paths.library_dir)
    for film in scan_library(library_dir):
        try:
            idx = _get_search_index(cfg, film.slug)
        except ValueError:
            continue
        if idx.status is IndexStatus.OK:
            return True
    return False
