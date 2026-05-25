"""Public ``find`` verb — dispatches to clip / bm25 / hybrid for one film.

The dispatcher wraps the legacy per-mode functions
(:func:`cinemateca.search.clip.search_text`,
:func:`cinemateca.search.clip.search_image`,
:func:`cinemateca.search.hybrid.search_hybrid`) behind the typed public
API locked in the P1 spec:

  >>> result = search.find(search.Query.text("man on a horse"), film=ctx)
  >>> result.hits[0].scene_id
  1

Design notes:

  * Image queries (``query.image_path is not None``) are CLIP-only. The
    ``mode`` argument is ignored — there is no hybrid/BM25 image search
    in P1 (BM25 has no image input). The returned ``SearchResult`` reports
    ``mode="clip"`` to match what actually ran.
  * Missing or corrupt CLIP index → ``SearchResult(hits=[], no_index=True)``.
    The caller (the slim route in T15) renders the empty-state HTML.
  * BM25 tunables (``stopwords_lang``, ``k1``, ``b``) are resolved lazily
    via :func:`api.deps.get_config`. When the app config isn't wired up
    (unit tests in isolation), the resolver falls back to the same
    defaults the legacy ``api/services/search._get_bm25_index_for_ctx``
    used (``None / 1.5 / 0.75``) — verbatim parity with the prior path.

Module hygiene: this module imports :func:`api.services.catalog.load_tag_index`
when ``filters.tags`` is non-empty. The carve-out is recorded in
``.importlinter`` (T11 already carved
``cinemateca.search.aggregate -> api.services.catalog``; T13 adds the
same line for ``_dispatch``). P2 will move ``load_tag_index`` under
``cinemateca.library`` and the carve-out deletes.
"""

from __future__ import annotations

from typing import Any

from cinemateca.search.cache import load_index
from cinemateca.search.clip import search_image
from cinemateca.search.hybrid import search_hybrid
from cinemateca.search.types import (
    Filters,
    Hit,
    HybridWeights,
    Query,
    SearchMode,
    SearchResult,
)

# Canonical filenames for the per-film CLIP index. Mirror the
# ``config/default.yaml`` → ``embeddings.*`` values and act as defaults
# when ``cfg.embeddings`` is absent (test configs that only supply
# ``paths.library_dir``).
_DEFAULT_EMBEDDINGS_FILENAME = "keyframe_embeddings.npy"
_DEFAULT_MAPPING_FILENAME = "index_mapping.json"


def find(
    query: Query,
    *,
    film: Any,
    mode: SearchMode = "hybrid",
    top_k: int = 20,
    filters: Filters | None = None,
    weights: HybridWeights | None = None,
) -> SearchResult:
    """Run a search against a single film.

    ``film`` is duck-typed: it must expose ``.slug``, ``.metadata_dir``
    and ``.embeddings_dir`` (the current producer is
    :class:`api.services.film_context.FilmContext`; P2 swaps in
    ``cinemateca.library.Library`` without changing the duck-typed
    surface).

    Image queries are CLIP-only — ``mode`` is ignored and forced to
    ``"clip"`` in the returned :class:`SearchResult`.
    """
    filters = filters or Filters()
    weights = weights or HybridWeights()

    index = load_index(
        film,
        embeddings_filename=_DEFAULT_EMBEDDINGS_FILENAME,
        mapping_filename=_DEFAULT_MAPPING_FILENAME,
    )
    if not index.ok:
        return SearchResult(
            hits=[],
            mode=mode,
            weights=weights if mode == "hybrid" else None,
            query=query,
            no_index=True,
        )

    if query.image_path is not None:
        df = search_image(index, query.image_path, top_k=top_k)
        return _df_to_result(df, mode="clip", weights=None, query=query)

    if query.text is None:
        raise ValueError("Query must have text or image_path set")

    tag_index = _load_tag_index(film) if filters.tags else {}
    bm25 = _load_bm25_for_mode(film, mode)

    df = search_hybrid(
        index,
        bm25=bm25,
        query=query.text,
        tags=list(filters.tags),
        tag_index=tag_index,
        top_k=top_k,
        min_similarity=filters.min_similarity,
        retriever_mode=mode,
        sem_w=weights.sem_w,
        bm25_w=weights.bm25_w,
        rrf_k=weights.rrf_k,
    )
    return _df_to_result(
        df,
        mode=mode,
        weights=weights if mode == "hybrid" else None,
        query=query,
    )


def _load_tag_index(film: Any) -> dict:
    """Lazy import to keep the module importable without ``api.*`` wired up."""
    from api.services.catalog import load_tag_index

    return load_tag_index(film.metadata_dir) or {}


def _load_bm25_for_mode(film: Any, mode: SearchMode):
    """Build the BM25 index for the film, or ``None`` for clip mode.

    Reads ``cfg.search.bm25`` tunables lazily; falls back to the same
    defaults the legacy ``_get_bm25_index_for_ctx`` used when no app
    config is wired up (unit-test isolation).
    """
    if mode == "clip":
        return None
    from cinemateca.search.bm25 import bm25_index_for_ctx

    stopwords_lang: str | None = None
    k1 = 1.5
    b = 0.75
    try:
        from api.deps import get_config

        cfg = get_config()
        bm25_cfg = getattr(cfg.search, "bm25", None) if hasattr(cfg, "search") else None
        if bm25_cfg is not None:
            stopwords_lang = getattr(bm25_cfg, "stopwords_lang", None)
            k1 = float(getattr(bm25_cfg, "k1", 1.5))
            b = float(getattr(bm25_cfg, "b", 0.75))
    except Exception:
        # No app config wired (unit-test isolation) — defaults match the
        # legacy ``_get_bm25_index_for_ctx`` fallback path byte-for-byte.
        pass

    return bm25_index_for_ctx(
        film,
        stopwords_lang=stopwords_lang,
        k1=k1,
        b=b,
    )


def _df_to_result(
    df, *, mode: SearchMode, weights: HybridWeights | None, query: Query
) -> SearchResult:
    """Convert a search DataFrame into a typed :class:`SearchResult`.

    The score column is ``similarity`` for every legacy producer (CLIP,
    BM25, fused). ``score`` is checked as a defensive fallback so a
    future producer that emits a different column name doesn't silently
    yield zeros.
    """
    hits: list[Hit] = []
    if df is not None and not df.empty:
        for row in df.to_dict("records"):
            score_val = row.get("similarity")
            if score_val is None:
                score_val = row.get("score", 0.0)
            hits.append(
                Hit(
                    scene_id=int(row["scene_id"]),
                    score=float(score_val),
                    keyframe_path=str(row.get("filepath", "")),
                )
            )
    return SearchResult(hits=hits, mode=mode, weights=weights, query=query)
