"""Search tab routes — text + image semantic search via CLIP. Thin HTTP
layer: parse, executor-offload, render. All retrieval logic lives in
:mod:`cinemateca.search` (P1); ``api/services/search.py`` re-exports
what tests + routes pin against and owns ``dispatch_text_search``.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Query, Request, UploadFile
from fastapi.responses import HTMLResponse

from api.deps import film_slug_query, get_config, make_ctx
from api.services import search as search_service
from api.services.catalog import derive_fps, load_json
from api.templates import templates
from cinemateca.library import FilmContext

logger = logging.getLogger(__name__)
router = APIRouter()


def build_search_context(slug: str | None = None) -> dict:
    """Re-exported for ``api/server.py``'s tab-context map."""
    cfg = get_config()
    if slug is not None:
        return search_service.build_search_context(FilmContext.for_film(cfg, slug), cfg)
    return search_service.build_search_context_aggregate(cfg)


def _kf_meta(ctx: FilmContext) -> tuple[dict, float]:
    kf_meta = load_json(ctx.metadata_dir / "keyframes_metadata.json") or []
    return {e["scene_id"]: e for e in kf_meta if "scene_id" in e}, derive_fps(kf_meta)


def _render(request: Request, **ctx) -> HTMLResponse:
    return templates.TemplateResponse(
        request, "partials/search_results.html", make_ctx(request, **ctx)
    )


def _no_index_response(request: Request) -> HTMLResponse:
    return _render(request, results=[], no_index=True, films_by_id={})


def _render_results(req: Request, *, slug, cfg, results, **extra) -> HTMLResponse:
    fbi = search_service.films_by_id_lookup(cfg)
    return _render(
        req, current_slug=slug, results=results, no_index=False, films_by_id=fbi, **extra
    )


def _enriched_per_film(cfg, ctx: FilmContext, results_df, slug: str | None) -> list[dict]:
    mbs, fps = _kf_meta(ctx)
    dicts = search_service.results_to_dicts(results_df, ctx.data_dir, mbs, fps)
    return search_service.enrich_hits_with_film_metadata(cfg, dicts, per_film_slug=slug)


@router.get("/tab/search", response_class=HTMLResponse)
async def tab_search(
    request: Request,
    slug: str | None = Depends(film_slug_query),
) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "partials/search.html",
        make_ctx(request, current_slug=slug, **build_search_context(slug)),
    )


@router.get("/api/search", response_class=HTMLResponse)
async def api_search(
    request: Request,
    q: str = "",
    tags: list[str] = Query(default=[]),
    top_k: int = 8,
    retriever: str = "hybrid",
    sem_w: float | None = None,
    bm25_w: float | None = None,
    modality: str = "text",
    w: float | None = None,
    reranker_enabled: bool | None = None,
    slug: str | None = Depends(film_slug_query),
) -> HTMLResponse:
    """Semantic search across one (``?film=<slug>``) or all films.

    ``modality`` selects the retrieval space (default ``"text"`` keeps
    the legacy CLIP/BM25/hybrid behaviour). ``"audio"`` routes through
    :mod:`cinemateca.search.audio` (CLAP joint text+audio space) — the
    rest of the text-only knobs (``tags``, ``retriever``, ``sem_w``,
    ``bm25_w``) are ignored on the audio path because CLAP doesn't
    expose them. ``"fusion"`` linearly combines CLIP keyframe + CLAP
    audio cosines under ``w`` (defaults to
    ``cfg.retrieval.fusion.visual_weight`` — fallback ``0.5``; clamped
    into ``[0, 1]`` for UX-friendliness over 422-rejecting).

    ``reranker_enabled`` controls the text-query cross-encoder reranker. The
    route applies it after result enrichment so the reranker can score the
    scene descriptions shown on cards.
    """
    q = q.strip()
    if len(q) < 2:
        return HTMLResponse("")
    cfg = get_config()
    if modality == "audio":
        return await _api_search_audio(request, q=q, top_k=top_k, slug=slug, cfg=cfg)
    if modality == "fusion":
        return await _api_search_fusion(request, q=q, top_k=top_k, w=w, slug=slug, cfg=cfg)
    min_sim = float(getattr(cfg.embeddings, "min_similarity", 0.0) or 0.0)
    retriever, sw, bw, rrf_k = search_service.resolve_retriever_args(cfg, retriever, sem_w, bm25_w)
    logger.info(
        f"api_search q={q!r} slug={slug or '(agg)'} retriever={retriever} top_k={top_k} "
        f"min_sim={min_sim:.3f} sw={sw:.3f} bw={bw:.3f} tags={list(tags) or None} "
        f"reranker_enabled={reranker_enabled}"
    )
    ctx = FilmContext.for_film(cfg, slug) if slug is not None else None
    args = (cfg, ctx, q, tags, top_k, min_sim, retriever, sw, bw, rrf_k)
    payload, no_index = await asyncio.get_running_loop().run_in_executor(
        None, lambda: search_service.dispatch_text_search(*args)
    )
    if no_index and (slug is not None or not payload):
        return _no_index_response(request)
    if slug is None:
        agg = search_service.aggregate_hits_to_template_dicts(cfg, payload) if payload else []
        results = search_service.enrich_hits_with_film_metadata(cfg, agg) if agg else []
    else:
        results = _enriched_per_film(cfg, ctx, payload, slug)
    results = search_service.rerank_template_results(
        results,
        cfg=cfg,
        query=q,
        mode=retriever,
        enabled=reranker_enabled,
    )
    return _render_results(
        request, slug=slug, cfg=cfg, results=results, query=q, highlighted_tags=set(tags)
    )


async def _api_search_audio(
    request: Request,
    *,
    q: str,
    top_k: int,
    slug: str | None,
    cfg,
) -> HTMLResponse:
    """Audio-only search dispatch (CLAP joint text+audio space).

    Thin shim: thread-pool offloads the encoder + dot-product into a
    worker thread (CLAP encode + matmul are blocking + non-trivial),
    then enriches the raw ``{scene_id, score}`` hits with the per-film
    template fields the ``.b-card`` partial expects (``img_url``,
    ``timecode``, ``similarity``, plus description / tags via the
    existing CLIP-side enrichment helper).
    """
    logger.info(f"api_search modality=audio q={q!r} slug={slug or '(agg)'} top_k={top_k}")
    ctx = FilmContext.for_film(cfg, slug) if slug is not None else None
    payload, no_index = await asyncio.get_running_loop().run_in_executor(
        None, lambda: search_service.dispatch_audio_search(cfg, ctx, q, top_k)
    )
    if no_index:
        return _no_index_response(request)
    card_dicts = search_service.audio_hits_to_template_dicts(cfg, payload, per_film_slug=slug)
    results = search_service.enrich_hits_with_film_metadata(cfg, card_dicts, per_film_slug=slug)
    return _render_results(request, slug=slug, cfg=cfg, results=results, query=q)


async def _api_search_fusion(
    request: Request,
    *,
    q: str,
    top_k: int,
    w: float | None,
    slug: str | None,
    cfg,
) -> HTMLResponse:
    """Cross-modal CLIP × CLAP fusion search.

    Linear late-fusion: ``score = w * clip_cosine + (1 - w) * clap_cosine``.
    ``w`` defaults to ``cfg.retrieval.fusion.visual_weight`` (0.5 unless
    overridden in local config). Clamped to ``[0, 1]`` — UX-friendly
    slider can briefly overshoot.

    Thin shim around :func:`search_service.dispatch_fusion_search`; the
    matmul + per-modality top-k go through the thread executor.
    """
    if w is None:
        fusion_cfg = getattr(getattr(cfg, "retrieval", None), "fusion", None)
        weight_raw = float(getattr(fusion_cfg, "visual_weight", 0.5) if fusion_cfg else 0.5)
    else:
        weight_raw = float(w)
    # Single clamp covers both branches — defends against malformed
    # local.yaml overrides that bypass the FusionConfig dataclass validator.
    weight = max(0.0, min(1.0, weight_raw))
    logger.info(
        f"api_search modality=fusion q={q!r} slug={slug or '(agg)'} "
        f"top_k={top_k} w={weight:.3f}"
    )
    ctx = FilmContext.for_film(cfg, slug) if slug is not None else None
    payload, no_index = await asyncio.get_running_loop().run_in_executor(
        None,
        lambda: search_service.dispatch_fusion_search(cfg, ctx, q, top_k, visual_weight=weight),
    )
    if no_index:
        return _no_index_response(request)
    # Fusion hits carry per-modality (clip_score, clap_score) alongside the
    # audio-hit shape (scene_id, score, film_slug, film_title). The audio
    # template converter constructs only the shape the b-card partial
    # renders today, so the per-modality scores are dropped at the template
    # boundary on purpose. Revisit when the card needs to show
    # "visual X.XX / audio Y.YY" — at that point widen the converter (or
    # create a fusion-specific one) instead of teaching the audio helper
    # about extra fields.
    card_dicts = search_service.audio_hits_to_template_dicts(cfg, payload, per_film_slug=slug)
    results = search_service.enrich_hits_with_film_metadata(cfg, card_dicts, per_film_slug=slug)
    return _render_results(request, slug=slug, cfg=cfg, results=results, query=q)


@router.post("/api/search/image", response_class=HTMLResponse)
async def api_search_image(
    request: Request,
    file: UploadFile = File(...),
    top_k: int = 8,
    slug: str | None = Depends(film_slug_query),
) -> HTMLResponse:
    """Image-similarity search.

    When ``?film=<slug>`` is supplied this searches that film's visual index;
    without a film it falls back to the legacy flat/global context rather than
    the aggregate multi-film dispatcher.
    """
    cfg = get_config()
    ctx = FilmContext.for_film(cfg, slug) if slug is not None else FilmContext.from_config(cfg)
    index = search_service.load_index(
        ctx,
        mapping_filename=cfg.embeddings.mapping_filename,
        embeddings_filename=cfg.embeddings.filename,
    )
    if not index.ok:
        return _no_index_response(request)
    # Read only up to the limit + 1 so we can size-check before buffering
    # the full payload into memory.  Files under the cap are read in full
    # in a single call; oversize files are rejected without a second read.
    data = await file.read(search_service.MAX_UPLOAD_BYTES + 1)
    if len(data) > search_service.MAX_UPLOAD_BYTES:
        msg = f"file too large ({len(data)} bytes > " f"{search_service.MAX_UPLOAD_BYTES} limit)"
        logger.info("Image-search upload rejected: %s", msg)
        return _render_results(request, slug=slug, cfg=cfg, results=[], upload_error=msg)
    try:
        suffix = search_service.validate_upload(file.filename, file.content_type, data)
    except search_service.UploadRejected as exc:
        logger.info("Image-search upload rejected: %s", exc)
        return _render_results(request, slug=slug, cfg=cfg, results=[], upload_error=str(exc))
    tmp_path: Path | None = None
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    try:
        loop = asyncio.get_running_loop()
        results_df = await loop.run_in_executor(
            None, search_service.search_image, index, tmp_path, top_k
        )
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
    results = _enriched_per_film(cfg, ctx, results_df, slug)
    return _render_results(request, slug=slug, cfg=cfg, results=results)
