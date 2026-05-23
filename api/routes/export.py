"""Structured catalog export routes.

HTTP concerns (Content-Type, Content-Disposition headers) live here, in
the route layer — intentionally NOT in the exporter helpers. The
exporters in ``cinemateca.exporters`` return plain strings/bytes only.
This separation means the exporter functions can be called from non-HTTP
contexts (CLI, tests) without dragging in FastAPI types.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from api.deps import get_config
from cinemateca.exporters import (
    ExportError,
    build_catalog_export,
    catalog_export_to_csv,
    catalog_export_to_json,
)

router = APIRouter()


def _export_or_404():
    try:
        return build_catalog_export(get_config())
    except ExportError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/api/export/catalog.json")
async def api_export_catalog_json() -> Response:
    """Return the current catalog as a reloadable JSON export."""

    text = catalog_export_to_json(_export_or_404())
    return Response(
        content=text,
        media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="catalog_export.json"'},
    )


@router.get("/api/export/catalog.csv")
async def api_export_catalog_csv() -> Response:
    """Return the current catalog as a flat CSV export."""

    text = catalog_export_to_csv(_export_or_404())
    return Response(
        content=text,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="catalog_export.csv"'},
    )
