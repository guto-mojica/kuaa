"""Annotate tab routes — manual scene tagging (thin HTTP layer).

Context builders live in :mod:`api.services.annotations` (A2 Task 5).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Form, Query, Request
from fastapi.responses import HTMLResponse

from api.deps import annotate_film_context, film_slug_query, make_ctx
from api.services.annotations import (
    build_annotate_context,
    build_description_edit_context,
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


def _scene_resp(request, slug, filter, tab, ctx, **extra) -> HTMLResponse:
    """Render ``partials/annotate_scene.html`` with the standard annotate context."""
    return templates.TemplateResponse(
        request,
        "partials/annotate_scene.html",
        make_ctx(request, current_slug=slug, filter=filter,
                 annotate_tab=normalize_annotate_tab(tab), **ctx, **extra),
    )


@router.get("/tab/annotate", response_class=HTMLResponse)
async def tab_annotate(
    request: Request,
    filter: str = Query(default="no_llm"),
    id: int | None = Query(default=None),
    tab: str = Query(default="comments"),
    slug: str | None = Depends(film_slug_query),
    fctx: FilmContext = Depends(annotate_film_context),
) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "partials/annotate.html",
        make_ctx(
            request, current_slug=slug, annotate_tab=normalize_annotate_tab(tab),
            **build_annotate_context(fctx, filter, id),
        ),
    )


@router.get("/api/annotate/scene", response_class=HTMLResponse)
async def api_annotate_scene(
    request: Request,
    id: int = Query(...),
    filter: str = Query(default="no_llm"),
    tab: str = Query(default="comments"),
    slug: str | None = Depends(film_slug_query),
    fctx: FilmContext = Depends(annotate_film_context),
) -> HTMLResponse:
    ctx = build_scene_panel(fctx, id, filter)
    return _scene_resp(request, slug, filter, tab, ctx)


@router.post("/api/annotate/save", response_class=HTMLResponse)
async def api_annotate_save(
    request: Request,
    scene_id: int = Form(...),
    filter: str = Form(default="no_llm"),
    tags: str = Form(default=""),
    tab: str = Form(default="annotations"),
    slug: str | None = Depends(film_slug_query),
    fctx: FilmContext = Depends(annotate_film_context),
) -> HTMLResponse:
    new_tags = normalize_tags(tags)
    ann = load_annotations(fctx)
    ann[str(scene_id)] = new_tags
    save_annotations(fctx, ann)
    logger.info("Saved %d tag(s) for scene %s", len(new_tags), scene_id)
    return _scene_resp(request, slug, filter, tab, build_scene_panel(fctx, scene_id, filter),
                       saved=True)


@router.get("/api/annotate/description/edit", response_class=HTMLResponse)
async def api_annotate_description_edit(
    request: Request,
    scene_id: int = Query(...),
    filter: str = Query(default="no_llm"),
    slug: str | None = Depends(film_slug_query),
    fctx: FilmContext = Depends(annotate_film_context),
) -> HTMLResponse:
    edit_ctx = build_description_edit_context(fctx, scene_id, filter)
    return templates.TemplateResponse(
        request,
        "partials/annotate_desc_edit.html",
        make_ctx(request, current_slug=slug, **edit_ctx),
    )


@router.post("/api/annotate/description", response_class=HTMLResponse)
async def api_annotate_description_save(
    request: Request,
    scene_id: int = Form(...),
    filter: str = Form(default="no_llm"),
    description: str = Form(default=""),
    tab: str = Form(default="comments"),
    slug: str | None = Depends(film_slug_query),
    fctx: FilmContext = Depends(annotate_film_context),
) -> HTMLResponse:
    save_description(fctx, scene_id, description.strip())
    logger.info("Description updated for scene %s", scene_id)
    return _scene_resp(request, slug, filter, tab, build_scene_panel(fctx, scene_id, filter),
                       desc_saved=True)


@router.post("/api/annotate/clear", response_class=HTMLResponse)
async def api_annotate_clear(
    request: Request,
    scene_id: int = Form(...),
    filter: str = Form(default="no_llm"),
    tab: str = Form(default="annotations"),
    slug: str | None = Depends(film_slug_query),
    fctx: FilmContext = Depends(annotate_film_context),
) -> HTMLResponse:
    ann = load_annotations(fctx)
    ann.pop(str(scene_id), None)
    save_annotations(fctx, ann)
    logger.info("Cleared tags for scene %s", scene_id)
    return _scene_resp(request, slug, filter, tab, build_scene_panel(fctx, scene_id, filter),
                       cleared=True)
