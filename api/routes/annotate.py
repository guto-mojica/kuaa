"""Annotate tab routes — manual scene tagging.

Thin HTTP layer (Phase 3b): every route here only parses the request,
delegates all JSON loading / id normalization / scene-list building /
persistence to ``api/services/annotations.py``, and renders. No
catalog/annotation logic is duplicated in this module anymore.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Form, Query, Request
from fastapi.responses import HTMLResponse

from api.deps import get_config, make_ctx
from api.services.annotations import (
    build_annotate_context,
    build_scene_panel,
    load_annotations,
    normalize_tags,
    save_annotations,
)
from api.services.film_context import FilmContext
from api.templates import templates

logger = logging.getLogger(__name__)
router = APIRouter()


def _ctx() -> FilmContext:
    return FilmContext.from_config(get_config())


@router.get("/tab/annotate", response_class=HTMLResponse)
async def tab_annotate(
    request: Request,
    filter: str = Query(default="no_llm"),
    id: int | None = Query(default=None),
) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "partials/annotate.html",
        make_ctx(request, **build_annotate_context(_ctx(), filter, id)),
    )


@router.get("/api/annotate/scene", response_class=HTMLResponse)
async def api_annotate_scene(
    request: Request,
    id: int = Query(...),
    filter: str = Query(default="no_llm"),
) -> HTMLResponse:
    ctx = build_scene_panel(_ctx(), id, filter)
    return templates.TemplateResponse(
        request,
        "partials/annotate_scene.html",
        make_ctx(request, filter=filter, **ctx),
    )


@router.post("/api/annotate/save", response_class=HTMLResponse)
async def api_annotate_save(
    request: Request,
    scene_id: int = Form(...),
    filter: str = Form(default="no_llm"),
    tags: str = Form(default=""),
) -> HTMLResponse:
    fctx = _ctx()

    new_tags = normalize_tags(tags)
    ann = load_annotations(fctx)
    ann[str(scene_id)] = new_tags
    save_annotations(fctx, ann)
    logger.info("Saved %d tag(s) for scene %s", len(new_tags), scene_id)

    ctx = build_scene_panel(fctx, scene_id, filter)
    return templates.TemplateResponse(
        request,
        "partials/annotate_scene.html",
        make_ctx(request, filter=filter, saved=True, **ctx),
    )


@router.post("/api/annotate/clear", response_class=HTMLResponse)
async def api_annotate_clear(
    request: Request,
    scene_id: int = Form(...),
    filter: str = Form(default="no_llm"),
) -> HTMLResponse:
    fctx = _ctx()

    ann = load_annotations(fctx)
    ann.pop(str(scene_id), None)
    save_annotations(fctx, ann)
    logger.info("Cleared tags for scene %s", scene_id)

    ctx = build_scene_panel(fctx, scene_id, filter)
    return templates.TemplateResponse(
        request,
        "partials/annotate_scene.html",
        make_ctx(request, filter=filter, cleared=True, **ctx),
    )
