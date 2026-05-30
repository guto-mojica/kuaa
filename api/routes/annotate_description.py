"""Annotate description-editing routes — inline edit / cancel / save.

Split from ``api/routes/annotate.py`` so the annotate core stays a thin
router within its LOC cap. The read-only Moondream description gets an
inline editor (open / abort / persist); persistence flows through
``save_description`` (atomic write). The abort route returns an empty
fragment so cancelling never re-renders or re-resolves the scene panel
(see the cancel docstring).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Form, Query, Request
from fastapi.responses import HTMLResponse

from api.deps import annotate_film_context, film_slug_query, make_ctx
from api.services.annotations import (
    build_description_edit_context,
    build_scene_panel,
    normalize_annotate_tab,
    save_description,
)
from api.templates import templates
from cinemateca.library import FilmContext

logger = logging.getLogger(__name__)
router = APIRouter()


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


@router.get("/api/annotate/description/cancel", response_class=HTMLResponse)
async def api_annotate_description_cancel() -> HTMLResponse:
    """Abort the inline description editor — clear the editor container only.

    Aborting an edit must NOT re-render the scene panel. The earlier Cancel
    wiring re-fetched ``/api/annotate/scene`` (a full panel re-render), which
    re-resolved the film/filter/scene context from scratch; whenever that
    resolution landed on a different scene (filter excluded the edited scene,
    or the active-film cookie disagreed with the omitted ``&film=``) the
    visible description vanished until the user navigated away and back. The
    editor lives in its own ``#annotate-llm-edit`` swap container *below* the
    read-only description, so returning an empty fragment closes the editor
    and leaves the already-correct description untouched.
    """
    return HTMLResponse("")


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
