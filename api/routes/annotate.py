"""Annotate tab routes — manual scene tagging.

Thin HTTP layer (Phase 3b): every route here only parses the request,
delegates all JSON loading / id normalization / scene-list building /
persistence to ``api/services/annotations.py``, and renders. No
catalog/annotation logic is duplicated in this module anymore.

T9: Routes accept an optional ``?film=<slug>`` query parameter.
``slug=None`` → flat ``FilmContext.from_config`` (single-film /
aggregate back-compat); ``slug=<value>`` → per-film
``FilmContext.for_film``.  An aggregate-annotate view is deferred to
a later plan; for now the annotate routes always operate on a single
film context.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Form, Query, Request
from fastapi.responses import HTMLResponse

from api.deps import film_ctx, film_slug_query, get_config, make_ctx
from api.services.annotations import (
    build_annotate_context,
    build_scene_panel,
    load_annotations,
    normalize_annotate_tab,
    normalize_tags,
    save_annotations,
    save_description,
)
from api.templates import templates
from cinemateca.library import FilmContext

logger = logging.getLogger(__name__)
router = APIRouter()


def _ctx(slug: str | None, request: Request | None = None) -> FilmContext:
    cfg = get_config()
    if slug is not None:
        return FilmContext.for_film(cfg, slug)
    if request is not None:
        return film_ctx(request, cfg)
    return FilmContext.from_config(cfg)


@router.get("/tab/annotate", response_class=HTMLResponse)
async def tab_annotate(
    request: Request,
    filter: str = Query(default="no_llm"),
    id: int | None = Query(default=None),
    tab: str = Query(default="comments"),
    slug: str | None = Depends(film_slug_query),
) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "partials/annotate.html",
        make_ctx(
            request,
            current_slug=slug,
            annotate_tab=normalize_annotate_tab(tab),
            **build_annotate_context(_ctx(slug, request), filter, id),
        ),
    )


@router.get("/api/annotate/scene", response_class=HTMLResponse)
async def api_annotate_scene(
    request: Request,
    id: int = Query(...),
    filter: str = Query(default="no_llm"),
    tab: str = Query(default="comments"),
    slug: str | None = Depends(film_slug_query),
) -> HTMLResponse:
    ctx = build_scene_panel(_ctx(slug, request), id, filter)
    return templates.TemplateResponse(
        request,
        "partials/annotate_scene.html",
        make_ctx(
            request,
            current_slug=slug,
            filter=filter,
            annotate_tab=normalize_annotate_tab(tab),
            **ctx,
        ),
    )


@router.post("/api/annotate/save", response_class=HTMLResponse)
async def api_annotate_save(
    request: Request,
    scene_id: int = Form(...),
    filter: str = Form(default="no_llm"),
    tags: str = Form(default=""),
    tab: str = Form(default="annotations"),
    slug: str | None = Depends(film_slug_query),
) -> HTMLResponse:
    fctx = _ctx(slug, request)

    new_tags = normalize_tags(tags)
    ann = load_annotations(fctx)
    ann[str(scene_id)] = new_tags
    save_annotations(fctx, ann)
    logger.info("Saved %d tag(s) for scene %s", len(new_tags), scene_id)

    ctx = build_scene_panel(fctx, scene_id, filter)
    return templates.TemplateResponse(
        request,
        "partials/annotate_scene.html",
        make_ctx(
            request,
            current_slug=slug,
            filter=filter,
            annotate_tab=normalize_annotate_tab(tab),
            saved=True,
            **ctx,
        ),
    )


@router.get("/api/annotate/description/edit", response_class=HTMLResponse)
async def api_annotate_description_edit(
    request: Request,
    scene_id: int = Query(...),
    filter: str = Query(default="no_llm"),
    slug: str | None = Depends(film_slug_query),
) -> HTMLResponse:
    from api.services.catalog import load_json

    fctx = _ctx(slug, request)
    descriptions = load_json(fctx.metadata_dir / "scene_descriptions.json") or []
    current = next(
        (d.get("description", "") for d in descriptions if d.get("scene_id") == scene_id),
        "",
    )
    return templates.TemplateResponse(
        request,
        "partials/annotate_desc_edit.html",
        make_ctx(
            request,
            current_slug=slug,
            scene_id=scene_id,
            filter=filter,
            current_description=current,
        ),
    )


@router.post("/api/annotate/description", response_class=HTMLResponse)
async def api_annotate_description_save(
    request: Request,
    scene_id: int = Form(...),
    filter: str = Form(default="no_llm"),
    description: str = Form(default=""),
    tab: str = Form(default="comments"),
    slug: str | None = Depends(film_slug_query),
) -> HTMLResponse:
    fctx = _ctx(slug, request)
    save_description(fctx, scene_id, description.strip())
    logger.info("Description updated for scene %s", scene_id)

    ctx = build_scene_panel(fctx, scene_id, filter)
    return templates.TemplateResponse(
        request,
        "partials/annotate_scene.html",
        make_ctx(
            request,
            current_slug=slug,
            filter=filter,
            annotate_tab=normalize_annotate_tab(tab),
            desc_saved=True,
            **ctx,
        ),
    )


@router.post("/api/annotate/clear", response_class=HTMLResponse)
async def api_annotate_clear(
    request: Request,
    scene_id: int = Form(...),
    filter: str = Form(default="no_llm"),
    tab: str = Form(default="annotations"),
    slug: str | None = Depends(film_slug_query),
) -> HTMLResponse:
    fctx = _ctx(slug, request)

    ann = load_annotations(fctx)
    ann.pop(str(scene_id), None)
    save_annotations(fctx, ann)
    logger.info("Cleared tags for scene %s", scene_id)

    ctx = build_scene_panel(fctx, scene_id, filter)
    return templates.TemplateResponse(
        request,
        "partials/annotate_scene.html",
        make_ctx(
            request,
            current_slug=slug,
            filter=filter,
            annotate_tab=normalize_annotate_tab(tab),
            cleared=True,
            **ctx,
        ),
    )
