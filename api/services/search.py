"""Search service — CLIP semantic-search domain logic + index validation.

This module owns what used to live inline in ``api/routes/search.py``:

  * loading the on-disk CLIP index (``.npy`` embeddings + the
    ``index_mapping.json`` keyframe map) — previously a private
    ``_load_index`` cached with ``@lru_cache(maxsize=1)`` keyed only on
    the directory path string, so a *regenerated* index was never picked
    up without a process restart and a stale/corrupt index leaked
    between requests;
  * result-DataFrame → template-dict conversion (``_results_to_dicts``);
  * the text / image search orchestration the route did inline.

Two correctness additions Phase 3c makes on top of the pure extraction
(catalog.py / annotations.py were byte-preserving refactors; this one is
explicitly a validation phase per the plan):

  1. **mtime/size-aware cache invalidation.** The cache is keyed by the
     embeddings + mapping file paths AND their ``(st_mtime_ns, st_size)``
     stat signature. A regenerated index (different size/mtime) is
     re-loaded automatically — no restart, no manual ``cache_clear``.
     Acknowledged blind spot: an index regenerated to a byte-identical
     ``st_size`` AND an identical ``st_mtime_ns`` would not be detected;
     this is practically impossible on a real regeneration (the writer
     touches mtime and the content/size changes), and a content hash is
     the only complete fix — deliberately not done, out of scope for a
     single-worker dev server.

  2. **Index shape validation.** ``CLIPEmbedder.load`` performs NO
     row-count consistency check (see its docstring — it just
     ``np.load`` + ``json.load``). A mapping that declares fewer
     keyframes than the embeddings matrix has rows previously crashed
     ``/api/search`` with a pandas ``IndexError`` (the Phase-2
     ``xfail(strict=True)`` tripwire). Validation lives HERE, in the
     api/services layer, deliberately: it keeps the AI core's contract
     (``embeddings.py``) untouched and does not change embedding/model
     computation or artefact formats — it only refuses to *serve* an
     incoherent index, degrading to a clear "corrupt index" UI state.

Path resolution flows through :class:`FilmContext` (consistent with the
catalog / annotations services). The RAW merged tag index is reused from
``api/services/catalog.load_tag_index`` and passed verbatim to
``SemanticSearch.combined`` — that method self-normalizes (Phase 1c), so
pre-normalizing here would diverge from the characterized contract.

Thread-safety: the module-level cache is guarded by a simple
``threading.Lock`` covering the stat-check + (re)load + store. The lock
is held across the disk load; that is acceptable for the current
single-worker dev server. Request-level concurrency / a job runner is
Phase 4 — this module deliberately does not over-engineer past a
correctness-preserving lock.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from api.services.catalog import (
    keyframe_url,  # noqa: F401  — re-exported for api.routes.search
)
from api.services.film_context import FilmContext

# Result conversion + Mojica context + films-by-id lookup — relocated to
# cinemateca.search._results and cinemateca.search._lookup (T8).
# Re-exported under the legacy names so external callers
# (``api/routes/search.py``, ``TestResultsToDicts``,
# ``test_multi_film_search.py``) keep working. ``_mojica_search_defaults``
# keeps its leading underscore here (it was private before T8); the new
# home publishes it as ``mojica_search_defaults`` for use within the
# search package.
from cinemateca.search._lookup import (
    build_search_context,  # noqa: F401
    build_search_context_aggregate,  # noqa: F401
    films_by_id_lookup,  # noqa: F401
)
from cinemateca.search._lookup import (
    mojica_search_defaults as _mojica_search_defaults,  # noqa: F401
)
from cinemateca.search._results import results_to_dicts  # noqa: F401

# BM25 loader + lru_cache — relocated to cinemateca.search.bm25.
# Re-exported under the legacy underscored names so external callers
# (``api/routes/search.py``, the existing tests) keep working. The
# module self-registers its cache flusher with
# :func:`cinemateca.search.cache.register_cache_clearer` at import time,
# so the wrapper around ``clear_index_cache`` that lived here under T6
# is gone — calling ``clear_index_cache()`` flushes BM25 transparently.
from cinemateca.search.bm25 import (
    _cached_bm25_index,  # noqa: F401  — legacy name for tests
    _file_stamp,  # noqa: F401  — legacy name for tests
    reindex_bm25,  # noqa: F401  — public P1 verb (T13 wires it into __init__)
)

# CLIP search-index loader + mtime/size cache — relocated to
# cinemateca.search.cache. Re-exported here so the legacy
# ``api.services.search.{IndexStatus, SearchIndex, load_index,
# clear_index_cache}`` import path keeps working for routes and the
# existing test suite (TestLoadIndexValidation, TestCacheInvalidation,
# TestFilmContextWiring). The ``_index_cache`` mapping is re-exported
# under its legacy name as well so tests that poke the dict directly
# (none today, but several reach in via ``cache_mod._index_cache`` in
# T6's own test file) continue to find it via either path.
from cinemateca.search.cache import (
    IndexStatus,  # noqa: F401
    SearchIndex,  # noqa: F401
    _index_cache,  # noqa: F401  — legacy name for tests that poke the dict
    clear_index_cache,  # noqa: F401  — flushes CLIP + BM25 via registered clearers
    load_index,  # noqa: F401
)

# CLIP search verbs — relocated to cinemateca.search.clip (T9). The
# names are re-exported here so external callers
# (``search_service.search_text`` / ``search_service.search_image``)
# and internal callers (``search_hybrid``) keep working unchanged.
from cinemateca.search.clip import (
    search_image,  # noqa: F401
    search_text,  # noqa: F401
)

# Degenerate-tag display filter — relocated to cinemateca.search.display.
# Re-exported under the legacy underscored names so external callers and
# the existing ``TestDegenerateTagFilter`` suite keep working.
from cinemateca.search.display import (
    filter_degenerate_tags as _filter_degenerate_tags,  # noqa: F401
)
from cinemateca.search.display import is_degenerate_tag as _is_degenerate_tag  # noqa: F401

# Upload validation — relocated to cinemateca.search.upload.
# UploadRejected lives in cinemateca.search.types (re-exported here so the
# existing ``api.services.search.UploadRejected`` import path keeps working
# for routes and the legacy ``TestValidateUpload`` suite).
from cinemateca.search.types import UploadRejected  # noqa: F401
from cinemateca.search.upload import (
    ALLOWED_IMAGE_SUFFIXES,  # noqa: F401
    MAX_UPLOAD_BYTES,  # noqa: F401
    validate_upload,  # noqa: F401
)

if TYPE_CHECKING:
    from cinemateca.retrieval.bm25 import BM25Index

logger = logging.getLogger(__name__)


# ── BM25 loader (relocated to cinemateca.search.bm25) ────────────────────────
#
# The core loader, its lru_cache, and the public ``reindex_bm25`` verb
# live in :mod:`cinemateca.search.bm25` (imported at the top of this
# file). This wrapper exists ONLY to resolve the BM25 tunables from the
# FastAPI app config — the core module is config-agnostic so it can be
# imported by tests / scripts without touching ``api.deps``.
def _get_bm25_index_for_ctx(ctx: FilmContext) -> BM25Index:
    """Load + cache the BM25 index for one film, using app-config tunables.

    Resolves ``cfg.search.bm25`` for ``stopwords_lang`` / ``k1`` / ``b``
    via :func:`api.deps.get_config`, then forwards to
    :func:`cinemateca.search.bm25.bm25_index_for_ctx`. ``get_config`` is
    imported lazily so the service module stays loadable without the
    FastAPI app config wired up (matters for unit tests that import
    this module in isolation).
    """
    from api.deps import get_config
    from cinemateca.search.bm25 import bm25_index_for_ctx

    cfg = get_config()
    bm25_cfg = getattr(cfg.search, "bm25", None)
    stopwords_lang = getattr(bm25_cfg, "stopwords_lang", None) if bm25_cfg else None
    k1 = float(getattr(bm25_cfg, "k1", 1.5)) if bm25_cfg else 1.5
    b = float(getattr(bm25_cfg, "b", 0.75)) if bm25_cfg else 0.75
    return bm25_index_for_ctx(
        ctx,
        stopwords_lang=stopwords_lang,
        k1=k1,
        b=b,
    )


# ── Per-film helpers + aggregate search ───────────────────────────────────────

# Canonical filenames for the per-film CLIP index.  These mirror the
# ``config/default.yaml`` → ``embeddings.*`` values and are used as
# defaults when ``cfg.embeddings`` is not present (e.g. in unit tests
# that supply a minimal SimpleNamespace config).
_DEFAULT_EMBEDDINGS_FILENAME = "keyframe_embeddings.npy"
_DEFAULT_MAPPING_FILENAME = "index_mapping.json"


def _get_embedder(cfg: Any) -> object:
    """Return a fresh ``OpenClipEmbedder`` instance.

    Extracted to module scope so unit tests can monkeypatch
    ``api.services.search._get_embedder`` to avoid loading the real CLIP
    model. The ``cfg`` argument is accepted for API consistency and future
    use (e.g. routing to a different backend via cfg) but is currently
    ignored.
    """
    from cinemateca.models.clip.openclip import OpenClipEmbedder

    return OpenClipEmbedder()


def _get_search_index(cfg: Any, slug: str) -> SearchIndex:
    """Return the (cached) :class:`SearchIndex` for the film identified by *slug*.

    Resolves the per-film embeddings directory via
    :meth:`FilmContext.for_film`, then delegates to :func:`load_index`
    with the canonical filenames (read from ``cfg.embeddings`` when
    available, falling back to the constants above for test configs that
    only supply ``paths.library_dir``).
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

      * library has no indexed films yet → render "No search index found"
        (user needs to run the embeddings pipeline);
      * library has indexed films but the query produced zero hits above
        ``min_similarity`` → render "No results" (the query simply didn't
        match anything in the corpus — a normal outcome, not a setup error).
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


# ── Aggregate cross-film search (relocated to cinemateca.search.aggregate) ────
# The aggregate orchestrator was extracted in T11. It still consumes the
# ``_get_embedder`` and ``_get_search_index`` helpers defined above via lazy
# attribute reads on this module, so any monkeypatch on
# ``api.services.search._get_{embedder,search_index}`` keeps hitting the
# call path inside the new module. Routes and tests that import
# ``api.services.search.aggregate_search`` continue to work via this
# re-export. A typed ``aggregate()`` wrapper lands in T13 behind the public
# ``cinemateca.search`` surface.
from cinemateca.search.aggregate import aggregate_search  # noqa: F401,E402

# ── Hybrid search dispatch (relocated to cinemateca.search.hybrid) ───────────
# CLIP verbs (search_text / search_image) live in cinemateca.search.clip (T9).
# The hybrid dispatcher + its 5 private helpers were extracted to
# cinemateca.search.hybrid in T10. Re-exported here under the legacy name so
# the route layer (``api/routes/search.py``) and the 12 M2 service tests
# (``tests/test_search_hybrid_service.py``) keep working unchanged.
#
# A signature reshape (``query``/``film``/``mode`` form, ``metadata_dir`` in
# place of a pre-loaded ``bm25``) lands in T13 behind the public
# ``cinemateca.search.find()`` verb. Verbatim move first, signature reshape
# behind a stable public surface second.
from cinemateca.search.hybrid import search_hybrid  # noqa: F401,E402

# ``search_image`` relocated to cinemateca.search.clip (T9) — re-exported above.
