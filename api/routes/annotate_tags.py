"""Annotate tag-curation routes — per-tag delete / rename + AI-tag suppression.

Split from ``api/routes/annotate.py``: the v0.8-rc annotations-as-retrieval
feature (commit 0c7f12d) added manual-tag delete/rename and non-destructive
AI-tag suppression. Each renders the scene panel after mutating the
annotations / ``tag_overrides.json`` layer. Kept in a focused router so
``annotate.py`` stays a thin core within its LOC cap; the service logic lives
in :mod:`api.services._annotate_curation` (re-exported on
``api.services.annotations``).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse

from api.deps import annotate_film_context, film_slug_query, make_ctx
from api.services.annotations import (
    build_scene_panel,
    delete_manual_tag,
    normalize_annotate_tab,
    rename_manual_tag,
    toggle_ai_tag,
)
from api.templates import templates
from cinemateca.library import FilmContext

logger = logging.getLogger(__name__)
router = APIRouter()


def _saved_scene(request: Request, slug: str | None, filter: str, tab: str, ctx: dict) -> HTMLResponse:
    """Render ``partials/annotate_scene.html`` (saved=True) after a curation edit."""
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


@router.post("/api/annotate/tag/delete", response_class=HTMLResponse)
async def api_annotate_tag_delete(
    request: Request,
    scene_id: int = Form(...),
    tag: str = Form(...),
    filter: str = Form(default="no_llm"),
    tab: str = Form(default="annotations"),
    slug: str | None = Depends(film_slug_query),
    fctx: FilmContext = Depends(annotate_film_context),
) -> HTMLResponse:
    delete_manual_tag(fctx, scene_id, tag)
    return _saved_scene(request, slug, filter, tab, build_scene_panel(fctx, scene_id, filter))


@router.post("/api/annotate/tag/edit", response_class=HTMLResponse)
async def api_annotate_tag_edit(
    request: Request,
    scene_id: int = Form(...),
    old_tag: str = Form(...),
    new_tag: str = Form(default=""),
    filter: str = Form(default="no_llm"),
    tab: str = Form(default="annotations"),
    slug: str | None = Depends(film_slug_query),
    fctx: FilmContext = Depends(annotate_film_context),
) -> HTMLResponse:
    rename_manual_tag(fctx, scene_id, old_tag, new_tag)
    return _saved_scene(request, slug, filter, tab, build_scene_panel(fctx, scene_id, filter))


@router.post("/api/annotate/ai-tag/toggle", response_class=HTMLResponse)
async def api_annotate_ai_tag_toggle(
    request: Request,
    scene_id: int = Form(...),
    tag: str = Form(...),
    suppressed: bool = Form(default=True),
    filter: str = Form(default="no_llm"),
    tab: str = Form(default="annotations"),
    slug: str | None = Depends(film_slug_query),
    fctx: FilmContext = Depends(annotate_film_context),
) -> HTMLResponse:
    """Suppress / restore a single AI-generated tag (non-destructive override).

    Writes ``tag_overrides.json`` only; ``scene_tags.json`` is untouched. The
    BM25 cache keys on the override file so the suppression takes effect on the
    next search.
    """
    toggle_ai_tag(fctx, scene_id, tag, suppressed=suppressed)
    return _saved_scene(request, slug, filter, tab, build_scene_panel(fctx, scene_id, filter))
