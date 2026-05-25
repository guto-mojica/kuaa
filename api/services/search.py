"""Search service — thin HTTP adapter over :mod:`cinemateca.search`.

After P1's deep-modules refactor (T3–T13) the search domain logic lives
in :mod:`cinemateca.search`. This module is the HTTP-adapter surface the
route layer + legacy test suite import through. It re-exports the
symbols the FastAPI app and tests pin against, plus a few small wrappers
that need the FastAPI app config (``_get_embedder`` monkeypatched by
tests; ``_get_search_index`` / ``_get_bm25_index_for_ctx`` resolve
``cfg.embeddings`` / ``cfg.search.bm25`` and forward to
:mod:`cinemateca.search`; ``has_indexed_films`` probes the library;
``dispatch_text_search`` orchestrates the per-film vs aggregate branch).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from api.services.catalog import keyframe_url  # noqa: F401  — used by routes
from cinemateca.library import FilmContext

# Result conversion + Mojica context + films-by-id lookup (T8). T15
# adds ``enrich_hits_with_film_metadata`` re-export so the slim route
# layer doesn't import ``cinemateca.search._lookup`` directly (keeps
# the routes-not-direct-core import-linter contract clean).
from cinemateca.search._lookup import (
    build_search_context,  # noqa: F401
    build_search_context_aggregate,  # noqa: F401
    enrich_hits_with_film_metadata,  # noqa: F401
    films_by_id_lookup,  # noqa: F401
)
from cinemateca.search._results import results_to_dicts  # noqa: F401

# Aggregate cross-film search (T11) — still reads ``_get_embedder`` and
# ``_get_search_index`` off this module via lazy attribute access, so the
# monkeypatches on ``api.services.search._get_*`` keep hitting the call path.
# T15 adds ``aggregate_hits_to_template_dicts`` re-export (same rationale
# as the ``enrich_hits_with_film_metadata`` re-export above).
from cinemateca.search.aggregate import (  # noqa: F401
    aggregate_hits_to_template_dicts,
    aggregate_search,
)

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

# Hybrid dispatch (T10). T15 adds ``resolve_retriever_args`` so the
# slim route imports HTTP-input normalisation from this layer rather
# than reaching into ``cinemateca.search.hybrid`` directly.
from cinemateca.search.hybrid import (  # noqa: F401
    resolve_retriever_args,
    search_hybrid,
)

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
    """Load + cache the BM25 index for one film. Resolves ``cfg.search.bm25``
    tunables (``stopwords_lang`` / ``k1`` / ``b``) via lazy ``get_config``
    so this module stays loadable without the FastAPI app wired up.
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
    """Return a fresh ``OpenClipEmbedder``. Module-scope so unit tests
    monkeypatch ``api.services.search._get_embedder`` to avoid loading
    the real CLIP model. ``cfg`` is accepted but currently ignored.
    """
    from cinemateca.models.clip.openclip import OpenClipEmbedder

    return OpenClipEmbedder()


def _get_search_index(cfg: Any, slug: str) -> SearchIndex:
    """Return the (cached) :class:`SearchIndex` for ``slug``. Reads
    ``cfg.embeddings.*`` filenames when present, otherwise falls back
    to the module-level defaults for minimal test configs.
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
        ctx, embeddings_filename=embeddings_filename, mapping_filename=mapping_filename
    )


def dispatch_text_search(
    cfg: Any,
    ctx: Any | None,
    q: str,
    tags: list[str],
    top_k: int,
    min_sim: float,
    retriever: str,
    sw: float,
    bw: float,
    rrf_k: int,
) -> tuple[Any, bool]:
    """Run text search; return ``(payload, no_index)``.

    ``ctx=None`` → cross-film aggregate (``payload`` = list of hit dicts).
    ``ctx`` given → per-film (``payload`` = a DataFrame). ``no_index=True``
    signals the route should render the no-index empty state instead of a
    results list. Pulls the BM25 index + tag index lazily so the per-film
    fast path doesn't read disk when ``retriever == "clip"`` and ``tags``
    is empty (preserving the legacy behaviour).
    """
    from api.services.catalog import load_tag_index

    if ctx is None:
        try:
            hits = aggregate_search(
                cfg,
                query=q,
                modality="text",
                top_k=top_k,
                tags=tags,
                min_similarity=min_sim,
                retriever_mode=retriever,
                sem_w=sw,
                bm25_w=bw,
                rrf_k=rrf_k,
            )
        except NotImplementedError:
            return [], True
        if not hits and not has_indexed_films(cfg):
            return [], True
        return hits, False

    index = load_index(
        ctx,
        mapping_filename=cfg.embeddings.mapping_filename,
        embeddings_filename=cfg.embeddings.filename,
    )
    if not index.ok:
        return None, True
    tag_index = load_tag_index(ctx.metadata_dir) if tags else {}
    if retriever == "clip":
        return search_text(index, q, tags, tag_index, top_k, min_sim), False
    bm25 = _get_bm25_index_for_ctx(ctx)
    return (
        search_hybrid(
            index,
            bm25=bm25,
            query=q,
            tags=tags,
            tag_index=tag_index,
            top_k=top_k,
            min_similarity=min_sim,
            retriever_mode=retriever,
            sem_w=sw,
            bm25_w=bw,
            rrf_k=rrf_k,
        ),
        False,
    )


def has_indexed_films(cfg: Any) -> bool:
    """``True`` iff at least one registered film has an OK :class:`SearchIndex`.

    Lets the route distinguish "no indexed films yet" (run the pipeline)
    from "indexed films exist but the query matched nothing" (no results).
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
