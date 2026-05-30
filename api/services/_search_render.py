"""Search render helpers extracted from ``api/routes/search.py`` (A2 / Task 5).

These functions handle result enrichment, template assembly, and the audio/
fusion dispatch shims. The route keeps only the FastAPI handlers and param
parsing; all rendering logic lives here.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from fastapi import Request
from fastapi.responses import HTMLResponse

from api.deps import make_ctx
from api.services import search as search_service
from api.services.catalog import derive_fps, load_json
from api.templates import templates
from cinemateca.library import FilmContext

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def kf_meta(ctx: FilmContext) -> tuple[dict, float]:
    """Return ``(scene_id→entry dict, fps)`` from keyframes_metadata.json."""
    kf = load_json(ctx.metadata_dir / "keyframes_metadata.json") or []
    return {e["scene_id"]: e for e in kf if "scene_id" in e}, derive_fps(kf)


def render_search(request: Request, **ctx) -> HTMLResponse:
    """Return the ``partials/search_results.html`` template response."""
    return templates.TemplateResponse(
        request, "partials/search_results.html", make_ctx(request, **ctx)
    )


def no_index_response(request: Request) -> HTMLResponse:
    """Return the no-index empty-state response."""
    return render_search(request, results=[], no_index=True, films_by_id={})


def render_results(req: Request, *, slug, cfg, results, **extra) -> HTMLResponse:
    """Enrich and render the results partial."""
    fbi = search_service.films_by_id_lookup(cfg)
    return render_search(
        req, current_slug=slug, results=results, no_index=False, films_by_id=fbi, **extra
    )


def enriched_per_film(cfg, ctx: FilmContext, results_df, slug: str | None) -> list[dict]:
    """Build per-film enriched result dicts from a raw results dataframe."""
    mbs, fps = kf_meta(ctx)
    dicts = search_service.results_to_dicts(results_df, ctx.data_dir, mbs, fps)
    return search_service.enrich_hits_with_film_metadata(cfg, dicts, per_film_slug=slug)


async def api_search_audio(
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
    template fields the ``.b-card`` partial expects.
    """
    logger.info(f"api_search modality=audio q={q!r} slug={slug or '(agg)'} top_k={top_k}")
    ctx = FilmContext.for_film(cfg, slug) if slug is not None else None
    payload, no_index = await asyncio.get_running_loop().run_in_executor(
        None, lambda: search_service.dispatch_audio_search(cfg, ctx, q, top_k)
    )
    if no_index:
        return no_index_response(request)
    card_dicts = search_service.audio_hits_to_template_dicts(cfg, payload, per_film_slug=slug)
    results = search_service.enrich_hits_with_film_metadata(cfg, card_dicts, per_film_slug=slug)
    return render_results(request, slug=slug, cfg=cfg, results=results, query=q)


async def api_search_fusion(
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
    overridden in local config). Clamped to ``[0, 1]``.
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
        return no_index_response(request)
    card_dicts = search_service.audio_hits_to_template_dicts(cfg, payload, per_film_slug=slug)
    results = search_service.enrich_hits_with_film_metadata(cfg, card_dicts, per_film_slug=slug)
    return render_results(request, slug=slug, cfg=cfg, results=results, query=q)
