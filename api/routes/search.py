"""Search tab routes — text and image semantic search via CLIP."""
from __future__ import annotations

import asyncio
import json
import logging
import tempfile
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, File, Query, Request, UploadFile
from fastapi.responses import HTMLResponse

from api.deps import get_config, make_ctx
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


def _load_tag_index(metadata_dir: Path) -> dict:
    from cinemateca.annotator import load as load_annotations, merge_tag_index

    tags_path = metadata_dir / "scene_tags.json"
    llm_tags: dict = {}
    if tags_path.exists():
        with open(tags_path, encoding="utf-8") as f:
            llm_tags = json.load(f)
    annotations = load_annotations(metadata_dir)
    return merge_tag_index(llm_tags, annotations)


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


def _results_to_dicts(results_df, data_dir: Path) -> list[dict]:
    return [
        {**row, "img_url": _keyframe_url(str(row["filepath"]), data_dir)}
        for row in results_df.to_dict("records")
    ]


def build_search_context() -> dict:
    """Build the template context the search tab partial needs.

    Shared by the ``/tab/search`` HTMX fragment and the ``/search``
    full-page route so both render identical markup.
    """
    cfg = get_config()
    tag_index = _load_tag_index(Path(cfg.paths.metadata_dir))
    available_tags = sorted(tag_index.keys()) if tag_index else []
    return {"available_tags": available_tags}


@router.get("/tab/search", response_class=HTMLResponse)
async def tab_search(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "partials/search.html",
        make_ctx(request, **build_search_context()),
    )


@router.get("/api/search", response_class=HTMLResponse)
async def api_search(
    request: Request,
    q: str = "",
    tags: list[str] = Query(default=[]),
    top_k: int = 8,
) -> HTMLResponse:
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
            request,
            "partials/search_results.html",
            make_ctx(request, results=[], no_index=True),
        )

    from cinemateca.embeddings import SemanticSearch

    searcher = SemanticSearch(embeddings, kf_df, embedder)
    loop = asyncio.get_event_loop()
    data_dir = Path(cfg.paths.data_dir).resolve()

    if tags:
        tag_index = _load_tag_index(Path(cfg.paths.metadata_dir))
        results_df = await loop.run_in_executor(
            None, lambda: searcher.combined(q, tags, tag_index, top_k)
        )
    else:
        results_df = await loop.run_in_executor(None, searcher.by_text, q, top_k)

    return templates.TemplateResponse(
        request,
        "partials/search_results.html",
        make_ctx(request, results=_results_to_dicts(results_df, data_dir), no_index=False),
    )


@router.post("/api/search/image", response_class=HTMLResponse)
async def api_search_image(
    request: Request,
    file: UploadFile = File(...),
    top_k: int = 8,
) -> HTMLResponse:
    cfg = get_config()
    embeddings, kf_df, embedder = _load_index(
        str(cfg.paths.embeddings_dir),
        cfg.embeddings.mapping_filename,
        cfg.embeddings.filename,
    )

    if embeddings is None:
        return templates.TemplateResponse(
            request,
            "partials/search_results.html",
            make_ctx(request, results=[], no_index=True),
        )

    suffix = Path(file.filename or "img.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        from cinemateca.embeddings import SemanticSearch

        searcher = SemanticSearch(embeddings, kf_df, embedder)
        loop = asyncio.get_event_loop()
        results_df = await loop.run_in_executor(None, searcher.by_image, tmp_path, top_k)
    finally:
        tmp_path.unlink(missing_ok=True)

    data_dir = Path(cfg.paths.data_dir).resolve()
    return templates.TemplateResponse(
        request,
        "partials/search_results.html",
        make_ctx(request, results=_results_to_dicts(results_df, data_dir), no_index=False),
    )
