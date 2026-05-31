"""Search service — thin HTTP adapter over :mod:`cinemateca.search`.

After P1's deep-modules refactor the search domain logic lives in
:mod:`cinemateca.search`. This module is the HTTP-adapter surface the
route layer + tests import through. It re-exports the symbols the app
and tests pin against, plus wrappers that need the FastAPI config
(``_get_embedder`` monkeypatched by tests; ``_get_bm25_index_for_ctx``
resolves ``cfg.search.bm25``; ``dispatch_text_search`` orchestrates
per-film vs aggregate).

The reranker boundary (typed-``SearchResult`` rerank + the card↔result
projection helpers ``cards_to_result`` / ``result_to_cards``) lives in
``_search_rerank``. ``apply_reranker`` stays here so tests can monkeypatch
``api.services.search.search_rerank``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from api.contexts import SearchContext
from api.services.catalog import keyframe_url  # noqa: F401  — used by routes
from cinemateca.library import FilmContext

# Cross-encoder rerank verb (Task 3.1). Aliased so tests can monkeypatch
# ``api.services.search.search_rerank`` without bypassing the wrapper.
from cinemateca.search import rerank as search_rerank  # noqa: F401
from cinemateca.search._lookup import (
    build_search_context as _build_search_context_core,
)
from cinemateca.search._lookup import (
    build_search_context_aggregate as _build_search_context_aggregate_core,
)

# Result conversion + Mojica context + films-by-id lookup (T8). T15
# adds ``enrich_hits_with_film_metadata`` re-export so the slim route
# layer doesn't import ``cinemateca.search._lookup`` directly (keeps
# the routes-not-direct-core import-linter contract clean).
from cinemateca.search._lookup import (
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
    _get_embedder,
    _get_search_index,
    aggregate_hits_to_template_dicts,
    aggregate_search,
    has_indexed_films,
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
from cinemateca.search.types import (  # noqa: F401
    Hit,
    Query,
    SearchMode,
    SearchResult,
    UploadRejected,
)
from cinemateca.search.upload import (
    MAX_UPLOAD_BYTES,  # noqa: F401
    validate_upload,  # noqa: F401
)

if TYPE_CHECKING:
    from cinemateca.retrieval.bm25 import BM25Index

# ── Re-exports from split sibling modules ────────────────────────────────────
# Routes import the reranker boundary as ``search_service.apply_reranker`` etc.;
# the names must remain resolvable on THIS module's surface.
from api.services._search_rerank import (  # noqa: F401
    _gpu_available,
    _reranker_settings,
    apply_reranker,
    cards_to_result,
    rerank_search_result,
    reranker_default_enabled,
    result_to_cards,
)

logger = logging.getLogger(__name__)


def _get_bm25_index_for_ctx(ctx: FilmContext) -> BM25Index:
    """Load + cache the BM25 index for one film. Resolves ``cfg.search.bm25``
    tunables (``stopwords_lang`` / ``k1`` / ``b`` / ``tokenizer`` / ``tag_boost``)
    via lazy ``get_config`` so this module stays loadable without the FastAPI
    app wired up.
    """
    from api.deps import get_config
    from cinemateca.search.bm25 import bm25_index_for_ctx

    cfg = get_config()
    bm25_cfg = getattr(cfg.search, "bm25", None)
    stopwords_lang = getattr(bm25_cfg, "stopwords_lang", None) if bm25_cfg else None
    k1 = float(getattr(bm25_cfg, "k1", 1.5)) if bm25_cfg else 1.5
    b = float(getattr(bm25_cfg, "b", 0.75)) if bm25_cfg else 0.75
    tokenizer_name = str(getattr(bm25_cfg, "tokenizer", "regex")) if bm25_cfg else "regex"
    tag_boost = int(getattr(bm25_cfg, "tag_boost", 1)) if bm25_cfg else 1
    return bm25_index_for_ctx(
        ctx,
        stopwords_lang=stopwords_lang,
        k1=k1,
        b=b,
        tokenizer_name=tokenizer_name,
        tag_boost=tag_boost,
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

    Reranking is applied downstream by the render layer: the enriched
    cards are lifted to a typed ``SearchResult`` (``cards_to_result``),
    reranked typed (``rerank_search_result``), then projected back for the
    template (``result_to_cards``) — this verb returns the first-stage
    ranking only.
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
        cfg=cfg,
    )
    if not index.ok:
        return None, True
    tag_index = load_tag_index(ctx.metadata_dir) if tags else {}

    if retriever == "clip":
        result_df = search_text(index, q, tags, tag_index, top_k, min_sim)
    else:
        bm25 = _get_bm25_index_for_ctx(ctx)
        result_df = search_hybrid(
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
        )

    return result_df, False


# A10: typed wrappers — ``api.contexts.SearchContext`` can't be imported in
# ``cinemateca.*`` (deep-modules rule), so we annotate at this api-layer boundary.


def build_search_context(ctx: Any, cfg: Any | None = None) -> SearchContext:
    """Typed wrapper over ``cinemateca.search._lookup.build_search_context``."""
    return _build_search_context_core(ctx, cfg)  # type: ignore[return-value]


def build_search_context_aggregate(cfg: Any) -> SearchContext:
    """Typed wrapper over ``cinemateca.search._lookup.build_search_context_aggregate``."""
    return _build_search_context_aggregate_core(cfg)  # type: ignore[return-value]
