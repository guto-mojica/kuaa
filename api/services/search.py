"""Search service — thin re-exporter (O-22 split).

The implementation was split into two focused sub-modules:

  * ``_search_text``  — index loading/cache, tag filtering, text-search
    orchestration, aggregate search, context builders.
  * ``_search_image`` — upload validation and image-similarity search.

All public names are re-exported here so existing callers continue to
``from api.services.search import ...`` unchanged.
"""
from __future__ import annotations

# Re-export everything from the text-search sub-module.
from api.services._search_text import (
    IndexStatus,
    SearchIndex,
    _DEFAULT_EMBEDDINGS_FILENAME,
    _DEFAULT_MAPPING_FILENAME,
    _filter_degenerate_tags,
    _get_embedder,
    _get_search_index,
    _index_cache,
    _is_degenerate_tag,
    _load_and_validate,
    _mojica_search_defaults,
    _stat_sig,
    aggregate_search,
    build_search_context,
    build_search_context_aggregate,
    clear_index_cache,
    films_by_id_lookup,
    has_indexed_films,
    load_index,
    results_to_dicts,
    search_text,
)

# Re-export everything from the image-search sub-module.
from api.services._search_image import (
    ALLOWED_IMAGE_SUFFIXES,
    MAX_UPLOAD_BYTES,
    UploadRejected,
    search_image,
    validate_upload,
)

__all__ = [
    # text-search
    "IndexStatus",
    "SearchIndex",
    "_DEFAULT_EMBEDDINGS_FILENAME",
    "_DEFAULT_MAPPING_FILENAME",
    "_filter_degenerate_tags",
    "_get_embedder",
    "_get_search_index",
    "_index_cache",
    "_is_degenerate_tag",
    "_load_and_validate",
    "_mojica_search_defaults",
    "_stat_sig",
    "aggregate_search",
    "build_search_context",
    "build_search_context_aggregate",
    "clear_index_cache",
    "films_by_id_lookup",
    "has_indexed_films",
    "load_index",
    "results_to_dicts",
    "search_text",
    # image-search
    "ALLOWED_IMAGE_SUFFIXES",
    "MAX_UPLOAD_BYTES",
    "UploadRejected",
    "search_image",
    "validate_upload",
]
