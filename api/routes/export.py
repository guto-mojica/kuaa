"""Structured catalog export routes.

HTTP concerns (Content-Type, Content-Disposition headers) live here, in
the route layer — intentionally NOT in the exporter helpers. The
exporters in ``cinemateca.exporters`` return plain strings/bytes only.
This separation means the exporter functions can be called from non-HTTP
contexts (CLI, tests) without dragging in FastAPI types.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from api.deps import get_config
from cinemateca.exporters import (
    ExportError,
    SceneSlice,
    build_catalog_export,
    catalog_export_to_csv,
    catalog_export_to_json,
    scenes_to_edl,
)
from cinemateca.library import derive_fps, load_json

logger = logging.getLogger(__name__)

router = APIRouter()


def _export_or_404():
    try:
        return build_catalog_export(get_config())
    except ExportError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/api/export/catalog.json", summary="Export full catalog as JSON")
async def api_export_catalog_json() -> Response:
    text = catalog_export_to_json(_export_or_404())
    return Response(
        content=text,
        media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="catalog_export.json"'},
    )


@router.get("/api/export/catalog.csv", summary="Export full catalog as CSV")
async def api_export_catalog_csv() -> Response:
    text = catalog_export_to_csv(_export_or_404())
    return Response(
        content=text,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="catalog_export.csv"'},
    )


class _SceneRef(BaseModel):
    film_slug: str
    scene_id: int
    scene_slug: str = ""


class _EdlExportRequest(BaseModel):
    scenes: list[_SceneRef]
    title: str = "Cinemateca Export"


@router.post(
    "/api/export/scenes.edl",
    summary="Export selected scenes as CMX 3600 EDL",
    responses={200: {"content": {"text/plain": {}}, "description": "EDL file attachment"}},
)
async def api_export_scenes_edl(body: _EdlExportRequest) -> Response:
    """Build a CMX 3600 EDL from a client-supplied list of scene references."""

    cfg = get_config()
    library_root = Path(cfg.paths.library_dir)

    # Cache keyframes_metadata per film slug to avoid duplicate reads.
    kf_cache: dict[str, list] = {}
    fps_cache: dict[str, float] = {}

    def _kf_meta(slug: str) -> list:
        if slug not in kf_cache:
            p = library_root / slug / "metadata" / "keyframes_metadata.json"
            raw = load_json(p) if p.exists() else []
            kf_cache[slug] = raw if isinstance(raw, list) else []
            fps_cache[slug] = derive_fps(kf_cache[slug])
        return kf_cache[slug]

    slices: list[SceneSlice] = []
    for ref in body.scenes:
        meta = _kf_meta(ref.film_slug)
        entry = next(
            (e for e in meta if isinstance(e, dict) and e.get("scene_id") == ref.scene_id),
            None,
        )
        if entry is None:
            logger.warning(
                "EDL export: scene_id=%d not found in slug=%s", ref.scene_id, ref.film_slug
            )
            continue
        fps = fps_cache[ref.film_slug]
        slices.append(
            SceneSlice(
                scene_id=ref.scene_id,
                film_slug=ref.film_slug,
                film_title=ref.film_slug,
                slug=ref.scene_slug or f"scene_{ref.scene_id:04d}",
                start_time_s=float(entry.get("start_time_s") or 0.0),
                end_time_s=float(entry.get("end_time_s") or 0.0),
                fps=fps,
            )
        )

    if not slices:
        raise HTTPException(status_code=422, detail="No valid scenes found for EDL export.")

    edl_text = scenes_to_edl(slices, title=body.title)
    return Response(
        content=edl_text,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="selection.edl"'},
    )
