"""Public ``find`` verb — dispatches to clip / bm25 / hybrid for one film.

The dispatcher wraps the legacy per-mode functions
(:func:`kuaa.search.clip.search_text`,
:func:`kuaa.search.clip.search_image`,
:func:`kuaa.search.hybrid.search_hybrid`) behind the typed public
API locked in the P1 spec:

  >>> result = search.find(search.Query.of_text("man on a horse"), film=ctx)
  >>> result.hits[0].scene_id
  1

Design notes:

  * Image queries (``query.image_path is not None``) are CLIP-only. The
    ``mode`` argument is ignored — there is no hybrid/BM25 image search
    in P1 (BM25 has no image input). The returned ``SearchResult`` reports
    ``mode="clip"`` to match what actually ran.
  * Missing or corrupt CLIP index → ``SearchResult(hits=[], no_index=True)``.
    The caller (the slim route in T15) renders the empty-state HTML.
  * BM25 tunables (``stopwords_lang``, ``k1``, ``b``) are accepted as
    kwargs with defaults that match the legacy fallback path
    (``None / 1.5 / 0.75``). Callers that resolve ``cfg.search.bm25``
    should pass the values explicitly; callers without a config wired up
    (unit tests) continue to work unchanged with the defaults.

Module hygiene: this module has zero imports from ``api.*``. The only
lazy import is :func:`kuaa.library.load_tag_index` (loaded only
when ``filters.tags`` is non-empty) which lives in core.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from kuaa.config import Settings
from kuaa.search.cache import load_index
from kuaa.search.clip import search_image
from kuaa.search.hybrid import search_hybrid
from kuaa.search.rerank import rerank as _rerank
from kuaa.search.types import (
    Filters,
    Hit,
    HybridWeights,
    Query,
    SearchMode,
    SearchResult,
)
from kuaa.timing import timed

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
    # BM25 tunables — defaults mirror the legacy fallback (None / 1.5 / 0.75)
    # so callers without an explicit config keep identical behaviour.
    # Callers that resolve cfg.search.bm25 should pass the values explicitly.
    bm25_stopwords_lang: str | None = None,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
    # C5: typed reranker wiring — OFF by default pending WS-4 tuning evidence.
    # Flip default to True only after E2/E6 ablation confirms a metric delta.
    rerank: bool = False,
    rerank_model: str = "default",
    cfg: Settings | None = None,
) -> SearchResult:
    """Run a search against a single film.

    ``film`` is duck-typed: it must expose ``.slug``, ``.metadata_dir``
    and ``.embeddings_dir`` (the current producer is
    :class:`kuaa.library.FilmContext`).

    Image queries are CLIP-only — ``mode`` is ignored and forced to
    ``"clip"`` in the returned :class:`SearchResult`.

    ``bm25_stopwords_lang``, ``bm25_k1``, and ``bm25_b`` tune the BM25
    index.  The defaults (``None / 1.5 / 0.75``) match the prior lazy-
    config fallback path, so existing callers need no changes.

    ``cfg`` is forwarded to :func:`kuaa.search.cache.load_index`
    so the registry-configured text encoder (e.g. SigLIP-multilingual,
    1024-dim) is swapped in for the OpenClip default after a non-default
    ``models.image_embedder`` is resolved. Omitting ``cfg`` keeps the
    OpenClip default — correct only when the on-disk index is OpenClip.
    """
    filters = filters or Filters()
    weights = weights or HybridWeights()

    index = load_index(
        film,
        embeddings_filename=_DEFAULT_EMBEDDINGS_FILENAME,
        mapping_filename=_DEFAULT_MAPPING_FILENAME,
        cfg=cfg,
    )
    if not index.ok:
        return SearchResult(
            hits=[],
            mode=mode,
            weights=weights if mode == "hybrid" else None,
            query=query,
            no_index=True,
            retriever_mode=mode,
            num_films_searched=1,
        )

    if query.image_path is not None:
        with timed("find.image") as t:
            df = search_image(index, query.image_path, top_k=top_k)
        result = _df_to_result(df, mode="clip", weights=None, query=query)
        result = replace(
            result,
            retriever_mode="clip",
            fusion_used=False,
            num_films_searched=1,
            latency_ms=t.elapsed_ms,
        )
        if rerank:
            result = _rerank(result, model=rerank_model)
        return result

    if query.text is None:
        raise ValueError("Query must have text or image_path set")

    tag_index = _load_tag_index(film) if filters.tags else {}
    bm25 = _load_bm25_for_mode(
        film,
        mode,
        bm25_stopwords_lang=bm25_stopwords_lang,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
    )

    with timed("find.text") as t:
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
    result = _df_to_result(
        df,
        mode=mode,
        weights=weights if mode == "hybrid" else None,
        query=query,
    )
    result = replace(
        result,
        retriever_mode=mode,
        fusion_used=(mode == "hybrid"),
        num_films_searched=1,
        latency_ms=t.elapsed_ms,
    )
    if rerank:
        result = _attach_descriptions(result, film)
        result = _rerank(result, model=rerank_model)
    return result


def _attach_descriptions(result: SearchResult, film: Any) -> SearchResult:
    """Fill empty ``Hit.description`` from the film's ``scene_descriptions.json``.

    The CLIP index df carries only ``filepath`` + ``scene_id`` (the openclip
    loader), so hits built by :func:`_df_to_result` have an empty description.
    The cross-encoder reranker scores ``(query, description)`` pairs — without
    this it would score against ``""`` and produce a meaningless reordering
    (this is exactly what confounded the WS-4 rerank ablation). Loaded lazily
    and only when reranking; keyed by ``scene_id`` (first row wins). A missing
    ``metadata_dir`` or file leaves descriptions untouched, so callers that
    already populated them keep working unchanged.
    """
    metadata_dir = getattr(film, "metadata_dir", None)
    if metadata_dir is None or not result.hits:
        return result
    from kuaa.library import load_json

    raw = load_json(metadata_dir / "scene_descriptions.json") or []
    if not isinstance(raw, list):
        return result
    by_scene: dict[int, str] = {}
    for entry in raw:
        try:
            sid = int(entry.get("scene_id"))
        except (AttributeError, TypeError, ValueError):
            continue
        if sid not in by_scene:
            by_scene[sid] = str(entry.get("description") or "")
    hits = [
        (
            replace(h, description=by_scene[h.scene_id])
            if (not h.description) and h.scene_id in by_scene
            else h
        )
        for h in result.hits
    ]
    return replace(result, hits=hits)


def _load_tag_index(film: Any) -> dict:
    """Lazy import to keep the module importable without ``api.*`` wired up."""
    from kuaa.library import load_tag_index

    return load_tag_index(film.metadata_dir) or {}


def _load_bm25_for_mode(
    film: Any,
    mode: SearchMode,
    *,
    bm25_stopwords_lang: str | None = None,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
):
    """Build the BM25 index for the film, or ``None`` for clip mode.

    Tunables are received as kwargs (resolved by the caller from
    ``cfg.search.bm25`` when available).  Defaults match the legacy
    fallback path (``None / 1.5 / 0.75``) — no api.* import needed.
    """
    if mode == "clip":
        return None
    from kuaa.search.bm25 import bm25_index_for_ctx

    return bm25_index_for_ctx(
        film,
        stopwords_lang=bm25_stopwords_lang,
        k1=bm25_k1,
        b=bm25_b,
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
                    description=str(row.get("description", "")),
                )
            )
    return SearchResult(hits=hits, mode=mode, weights=weights, query=query)
