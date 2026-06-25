"""A5: liveness + readiness probes."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from api.deps import get_config
from api.schemas import HealthStatus, ReadyStatus

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthStatus)
async def health() -> HealthStatus:
    return HealthStatus(status="ok")


@router.get("/ready", response_model=ReadyStatus)
async def ready() -> JSONResponse:
    cfg = get_config()
    checks: dict[str, bool] = {}
    data_dir = Path(cfg.paths.data_dir)
    checks["data_dir_readable"] = data_dir.is_dir()
    library_dir = Path(cfg.paths.library_dir)
    try:
        from kuaa.library import load_registry

        load_registry(library_dir)
        checks["registry_parseable"] = True
    except Exception:
        checks["registry_parseable"] = False
    ok = all(checks.values())
    payload = ReadyStatus(ready=ok, checks=checks)
    return JSONResponse(payload.model_dump(), status_code=200 if ok else 503)
