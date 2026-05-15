"""Search tab routes — text and image semantic search via CLIP."""
from __future__ import annotations

import asyncio
import logging
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from api.deps import get_config
from api.templates import templates

logger = logging.getLogger(__name__)
router = APIRouter()


@lru_cache(maxsize=1)
def _load_index(embeddings_dir: str, mapping_filename: str, embeddings_filename: str):
    """Load CLIP embeddings from disk. Cached for the process lifetime."""
    from cinemateca.embeddings import CLIPEmbedder

    emb_path = Path(embeddings_dir) / embeddings_filename
    map_path = Path(embeddings_dir) / mapping_filename
    if not emb_path.exists() or not map_path.exists():
        logger.warning("Search index not found at %s", embeddings_dir)
        return None, None, None
    embeddings, _mapping, kf_df = CLIPEmbedder.load(emb_path, map_path)
    embedder = CLIPEmbedder()
    logger.info("Search index loaded: %d vectors", len(kf_df))
    return embeddings, kf_df, embedder


def _keyframe_url(filepath: str, data_dir: Path) -> str | None:
    """Convert a stored filepath to a /media/... URL."""
    fp = Path(filepath)
    for candidate in (fp, Path.cwd() / fp):
        try:
            rel = candidate.resolve().relative_to(data_dir.resolve())
            return f"/media/{rel.as_posix()}"
        except ValueError:
            continue
    return None


@router.get("/tab/search", response_class=HTMLResponse)
async def tab_search(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("partials/search.html", {"request": request})


@router.get("/api/search", response_class=HTMLResponse)
async def api_search(request: Request, q: str = "", top_k: int = 8) -> HTMLResponse:
    q = q.strip()
    if len(q) < 2:
        return HTMLResponse("")

    cfg = get_config()
    embeddings, kf_df, embedder = _load_index(
        str(cfg.paths.embeddings_dir),
        cfg.embeddings.mapping_filename,
        cfg.embeddings.filename,
    )

    if embeddings is None:
        return templates.TemplateResponse(
            "partials/search_results.html",
            {"request": request, "results": [], "no_index": True},
        )

    from cinemateca.embeddings import SemanticSearch

    searcher = SemanticSearch(embeddings, kf_df, embedder)
    loop = asyncio.get_event_loop()
    results_df = await loop.run_in_executor(None, searcher.by_text, q, top_k)

    data_dir = Path(cfg.paths.data_dir).resolve()
    results = [
        {**row, "img_url": _keyframe_url(str(row["filepath"]), data_dir)}
        for row in results_df.to_dict("records")
    ]

    return templates.TemplateResponse(
        "partials/search_results.html",
        {"request": request, "results": results, "no_index": False},
    )
