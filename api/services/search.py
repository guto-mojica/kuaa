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
from cinemateca.search.types import UploadRejected  # noqa: F401
from cinemateca.search.upload import (
    MAX_UPLOAD_BYTES,  # noqa: F401
    validate_upload,  # noqa: F401
)

if TYPE_CHECKING:
    from cinemateca.retrieval.bm25 import BM25Index

logger = logging.getLogger(__name__)


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


def dispatch_audio_search(
    cfg: Any,
    ctx: Any | None,
    q: str,
    top_k: int,
) -> tuple[list[dict], bool]:
    """Run audio (CLAP) search; return ``(hits, no_index)``.

    ``ctx=None`` → cross-film: walk every registered film, read its
    per-film CLAP index, run :func:`search_audio` against the same
    query vector, concatenate results, and take the global top-``top_k``
    by raw cosine. CLAP vectors are L2-normalised AND share a single
    joint text+audio space, so cosines are cross-film-comparable — no
    fusion / RRF reshape needed for parity with the CLIP path.

    ``ctx`` given → per-film: load ``<film_dir>/audio/`` only.

    ``no_index=True`` is returned when:
      * per-film: that film has no CLAP index on disk;
      * aggregate: no registered film has a CLAP index.

    The route renders the no-index empty state in both cases, same
    contract as the text path. The embedder is instantiated exactly
    once per dispatch (so the aggregate path doesn't reload CLAP per
    film); the call site monkeypatches
    ``cinemateca.models.registry.get_audio_embedder`` to skip the
    real model load in tests.
    """
    from cinemateca.library import scan_library
    from cinemateca.models.registry import get_audio_embedder
    from cinemateca.search.audio import load_audio_index, search_audio

    if ctx is not None:
        audio_dir = Path(ctx.metadata_dir).parent / "audio"
        index = load_audio_index(audio_dir)
        if index is None:
            return [], True
        embedder = get_audio_embedder(cfg, device=None)
        hits = search_audio(index, embedder, q, top_k=top_k)
        for h in hits:
            h["film_slug"] = ctx.slug
        return hits, False

    # Aggregate. Walk the registry, skip films without a CLAP index.
    library_dir = Path(cfg.paths.library_dir)
    films = list(scan_library(library_dir))
    if not films:
        return [], True
    embedder = None
    all_hits: list[dict] = []
    any_index = False
    for film in films:
        film_audio_dir = library_dir / film.slug / "audio"
        idx = load_audio_index(film_audio_dir)
        if idx is None:
            continue
        any_index = True
        if embedder is None:
            # Lazy-load the embedder ONCE after we know at least one
            # film has a CLAP index — keeps the empty-library / no-CLAP
            # case from paying the CLAP load cost.
            embedder = get_audio_embedder(cfg, device=None)
        film_hits = search_audio(idx, embedder, q, top_k=top_k)
        for h in film_hits:
            h["film_slug"] = film.slug
            h["film_title"] = film.title
            all_hits.append(h)
    if not any_index:
        return [], True
    all_hits.sort(key=lambda r: r["score"], reverse=True)
    return all_hits[:top_k], False


def audio_hits_to_template_dicts(
    cfg: Any, hits: list[dict], *, per_film_slug: str | None = None
) -> list[dict]:
    """Convert :func:`dispatch_audio_search` raw hits to template-card dicts.

    Mirrors :func:`aggregate_hits_to_template_dicts` (the CLIP-aggregate
    converter): aliases ``score → similarity`` so the same
    ``partials/search_results.html`` renders the CLAP path, resolves
    a per-scene ``keyframe_path`` + ``timecode`` from each film's
    ``keyframes_metadata.json`` so the card shows the visual proxy for
    the matched audio segment, and resolves ``img_url`` via
    :func:`cinemateca.library.keyframe_url`.

    Per-scene metadata lookup is memoised per slug — a 100-row result
    list reads each film's ``keyframes_metadata.json`` at most once.
    """
    from cinemateca.library import (
        FilmContext,
        derive_fps,
        load_json,
        to_smpte,
    )

    data_dir = Path(cfg.paths.data_dir).resolve()
    kf_cache: dict[str, tuple[dict, float]] = {}

    def _kf_for(slug: str) -> tuple[dict, float]:
        if slug in kf_cache:
            return kf_cache[slug]
        try:
            ctx = FilmContext.for_film(cfg, slug)
        except ValueError:
            kf_cache[slug] = ({}, 24.0)
            return kf_cache[slug]
        kf_meta = load_json(ctx.metadata_dir / "keyframes_metadata.json") or []
        by_scene = {int(e["scene_id"]): e for e in kf_meta if "scene_id" in e}
        kf_cache[slug] = (by_scene, derive_fps(kf_meta))
        return kf_cache[slug]

    out: list[dict] = []
    for h in hits:
        slug = h.get("film_slug") or per_film_slug or ""
        sid = int(h["scene_id"])
        by_scene, fps = _kf_for(slug) if slug else ({}, 24.0)
        meta = by_scene.get(sid) or {}
        kf_path = meta.get("filepath", "") or meta.get("keyframe_path", "") or ""
        start_s = float(meta.get("start_time_s") or 0.0)
        out.append(
            {
                "film_slug": slug,
                "scene_id": sid,
                "similarity": float(h["score"]),
                "img_url": keyframe_url(kf_path, data_dir) if kf_path else None,
                "timecode": to_smpte(start_s, fps) if start_s > 0 else "",
            }
        )
    return out


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
