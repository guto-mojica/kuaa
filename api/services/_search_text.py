"""Text-search functions — split from search.py (O-22).

Contains: index loading/cache, tag filter, text-search orchestration,
aggregate search, context builders, and shared result helpers.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from api.services.catalog import (
    derive_fps,
    keyframe_url,
    load_json,
    load_tag_index,
    to_smpte,
)
from api.services.film_service import list_films
from cinemateca.library import FilmContext

logger = logging.getLogger(__name__)

# ── Degenerate-tag display filter ─────────────────────────────────────────────
# scene_tags.json carries raw model-output fragments alongside the curated
# tag vocabulary — full captions, stuck-token repetitions, enumerated lists.
# They explode the visible tag-pill grid and add no signal (the long-tail
# entries cover 1–2 scenes each), so the displayed vocabulary drops them.
# Filtering is DISPLAY-ONLY: the underlying tag_index is unmodified, so a
# search request that arrives with a degenerate-looking ``tags=...`` query
# (manually crafted URL) still works on the per-film path.
_DEGENERATE_TAG_MAX_LEN = 20
_DEGENERATE_TAG_MAX_HYPHENS = 2
_REPEATED_TOKEN_RE = re.compile(r"\b(\w+)(?:[-\s]\1\b){2,}", re.IGNORECASE)
_TRAILING_NUMBER_RE = re.compile(r"-\d+$")
_DIGIT_LED_RE = re.compile(r"^\d+-")
_ARTICLE_LED_RE = re.compile(r"^(a|the)-", re.IGNORECASE)


def _is_degenerate_tag(tag: str) -> bool:
    """True when ``tag`` looks like raw model output, not a curated label."""
    if not tag:
        return True
    if tag.isdigit():
        return True
    if len(tag) > _DEGENERATE_TAG_MAX_LEN:
        return True
    if "." in tag.rstrip("."):
        return True
    if tag.endswith(".") and _ARTICLE_LED_RE.match(tag):
        return True
    if _REPEATED_TOKEN_RE.search(tag):
        return True
    if tag.count("-") > _DEGENERATE_TAG_MAX_HYPHENS:
        return True
    if _DIGIT_LED_RE.match(tag):
        return True
    if _TRAILING_NUMBER_RE.search(tag):
        return True
    return False


def _filter_degenerate_tags(tags) -> list[str]:
    """Drop degenerate-looking tag strings from the displayed vocabulary."""
    return [t for t in tags if not _is_degenerate_tag(t)]


class IndexStatus(str, Enum):
    """Outcome of attempting to load the search index."""

    OK = "ok"
    MISSING = "missing"  # .npy and/or mapping file absent
    CORRUPT = "corrupt"  # files present but shape-inconsistent / unreadable


@dataclass(frozen=True)
class SearchIndex:
    """A loaded (or failed) CLIP search index."""

    status: IndexStatus
    embeddings: object | None = None
    kf_df: object | None = None
    embedder: object | None = None
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status is IndexStatus.OK


# ── mtime/size-aware index cache ──────────────────────────────────────────────

_index_cache: dict[tuple[str | None, str, str], tuple[tuple, SearchIndex]] = {}
_cache_lock = threading.Lock()


def _stat_sig(path: Path) -> tuple[int, int] | None:
    """Return ``(st_mtime_ns, st_size)`` for *path*, or ``None`` if absent."""
    try:
        st = path.stat()
    except (FileNotFoundError, NotADirectoryError):
        return None
    return (st.st_mtime_ns, st.st_size)


def _load_and_validate(emb_path: Path, map_path: Path) -> SearchIndex:
    """Load the index from disk and validate its shape coherence."""
    if not emb_path.exists() or not map_path.exists():
        logger.warning("Search index not found at %s", emb_path.parent)
        return SearchIndex(IndexStatus.MISSING, detail="index files absent")

    from cinemateca.models.clip.openclip import OpenClipEmbedder

    try:
        embeddings, mapping, kf_df = OpenClipEmbedder.load(emb_path, map_path)
    except Exception as exc:
        logger.warning("Search index unreadable (%s): %s", emb_path.parent, exc)
        return SearchIndex(IndexStatus.CORRUPT, detail=f"unreadable: {exc}")

    n_emb = int(getattr(embeddings, "shape", [0])[0])
    n_map = len(kf_df)
    declared = mapping.get("total_vectors")
    if n_emb != n_map:
        logger.warning(
            "Corrupt search index: %d embedding rows vs %d keyframe-map rows (%s)",
            n_emb,
            n_map,
            emb_path.parent,
        )
        return SearchIndex(
            IndexStatus.CORRUPT,
            detail=f"row mismatch: {n_emb} embeddings vs {n_map} mapping rows",
        )
    if declared is not None and declared != n_map:
        logger.warning(
            "Corrupt search index: mapping declares total_vectors=%s but has %d keyframe rows (%s)",
            declared,
            n_map,
            emb_path.parent,
        )
        return SearchIndex(
            IndexStatus.CORRUPT,
            detail=(f"mapping total_vectors={declared} != {n_map} keyframe rows"),
        )

    embedder = OpenClipEmbedder()
    logger.info("Search index loaded: %d vectors", n_map)
    return SearchIndex(IndexStatus.OK, embeddings=embeddings, kf_df=kf_df, embedder=embedder)


def load_index(ctx: FilmContext, *, mapping_filename: str, embeddings_filename: str) -> SearchIndex:
    """Return the (cached) :class:`SearchIndex` for *ctx*'s embeddings dir."""
    emb_path = ctx.embeddings_dir / embeddings_filename
    map_path = ctx.embeddings_dir / mapping_filename
    key = (ctx.slug, str(emb_path), str(map_path))
    sig = (_stat_sig(emb_path), _stat_sig(map_path))

    with _cache_lock:
        cached = _index_cache.get(key)
        if cached is not None and cached[0] == sig:
            return cached[1]
        index = _load_and_validate(emb_path, map_path)
        _index_cache[key] = (sig, index)
        return index


def clear_index_cache() -> None:
    """Drop every cached index entry (test-isolation hook)."""
    with _cache_lock:
        _index_cache.clear()


# ── Per-film helpers + aggregate search ───────────────────────────────────────

_DEFAULT_EMBEDDINGS_FILENAME = "keyframe_embeddings.npy"
_DEFAULT_MAPPING_FILENAME = "index_mapping.json"


def _get_embedder(cfg: Any) -> object:
    """Return a fresh ``OpenClipEmbedder`` instance."""
    from cinemateca.models.clip.openclip import OpenClipEmbedder

    return OpenClipEmbedder()


def _get_search_index(cfg: Any, slug: str) -> SearchIndex:
    """Return the (cached) :class:`SearchIndex` for the film identified by *slug*."""
    emb_cfg = getattr(cfg, "embeddings", None)
    embeddings_filename = (
        getattr(emb_cfg, "filename", _DEFAULT_EMBEDDINGS_FILENAME)
        if emb_cfg is not None
        else _DEFAULT_EMBEDDINGS_FILENAME
    )
    mapping_filename = (
        getattr(emb_cfg, "mapping_filename", _DEFAULT_MAPPING_FILENAME)
        if emb_cfg is not None
        else _DEFAULT_MAPPING_FILENAME
    )
    ctx = FilmContext.for_film(cfg, slug)
    return load_index(
        ctx,
        embeddings_filename=embeddings_filename,
        mapping_filename=mapping_filename,
    )


def has_indexed_films(cfg: Any) -> bool:
    """``True`` iff at least one registered film has an OK :class:`SearchIndex`."""
    library_dir = Path(cfg.paths.library_dir)
    for film in list_films(library_dir):
        try:
            idx = _get_search_index(cfg, film.slug)
        except ValueError:
            continue
        if idx.status is IndexStatus.OK:
            return True
    return False


def aggregate_search(
    cfg: Any,
    *,
    query: str,
    modality: str,
    top_k: int,
    tags: list[str] | None = None,
    min_similarity: float = 0.0,
) -> list[dict]:
    """Run per-film search and merge top results by score."""
    from cinemateca.scene_ids import normalize_tag_index, scene_id_key

    if modality != "text":
        raise NotImplementedError(
            f"modality={modality!r} lands in a later plan; only 'text' is supported here"
        )

    library_dir = Path(cfg.paths.library_dir)
    films = list(list_films(library_dir))
    if not films:
        return []

    embedder = _get_embedder(cfg)

    text_vec: np.ndarray = embedder.encode_text(query)  # type: ignore[union-attr]
    norm = float(np.linalg.norm(text_vec))
    text_vec = text_vec / (norm + 1e-12)

    selected_tags = list(tags) if tags else []

    logger.info(
        "aggregate_search: query=%r films=%d top_k=%d tags=%s min_sim=%.3f",
        query,
        len(films),
        top_k,
        selected_tags or None,
        min_similarity,
    )

    all_hits: list[dict] = []
    per_film_kept = 0
    for film in films:
        try:
            idx = _get_search_index(cfg, film.slug)
        except ValueError as exc:
            logger.warning("aggregate_search: skip film %s — %s", film.slug, exc)
            continue
        if idx.status is not IndexStatus.OK:
            logger.info(
                "aggregate_search: skip film %s — index status %s",
                film.slug,
                idx.status,
            )
            continue
        ctx = FilmContext.for_film(cfg, film.slug)
        kf_meta = load_json(ctx.metadata_dir / "keyframes_metadata.json") or []
        fps = derive_fps(kf_meta)
        meta_by_scene = {e["scene_id"]: e for e in kf_meta if "scene_id" in e}

        allowed_scene_keys: set[str] | None = None
        if selected_tags:
            raw_index = load_tag_index(ctx.metadata_dir)
            norm_index = normalize_tag_index(raw_index)
            allowed_scene_keys = set(norm_index.get(selected_tags[0], set()))
            for t in selected_tags[1:]:
                allowed_scene_keys &= set(norm_index.get(t, set()))
            if not allowed_scene_keys:
                continue

        scores: np.ndarray = idx.embeddings @ text_vec  # type: ignore[operator]
        film_added = 0
        for i, score in enumerate(scores):
            if float(score) < min_similarity:
                continue
            row = idx.kf_df.iloc[i]  # type: ignore[union-attr]
            scene_id = int(row["scene_id"])
            if allowed_scene_keys is not None and scene_id_key(scene_id) not in allowed_scene_keys:
                continue
            meta = meta_by_scene.get(scene_id)
            start_s = float(meta.get("start_time_s") or 0.0) if meta else 0.0
            timecode = to_smpte(start_s, fps) if start_s > 0 else ""
            all_hits.append(
                {
                    "film_slug": film.slug,
                    "film_title": film.title,
                    "scene_id": scene_id,
                    "score": float(score),
                    "keyframe_path": str(row["filepath"]),
                    "timecode": timecode,
                }
            )
            film_added += 1
        if scores.size:
            top3 = np.sort(scores)[-3:][::-1]
            logger.info(
                "aggregate_search: film=%s n_vectors=%d top3=%s kept=%d",
                film.slug,
                int(scores.size),
                [round(float(s), 3) for s in top3],
                film_added,
            )
        per_film_kept += film_added

    all_hits.sort(key=lambda h: -h["score"])
    seen: set[tuple[str, int]] = set()
    deduped: list[dict] = []
    for h in all_hits:
        key = (h["film_slug"], h["scene_id"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(h)
    result = deduped[:top_k]
    logger.info(
        "aggregate_search: query=%r raw_kept=%d dedup_kept=%d returned=%d top_score=%.3f",
        query,
        per_film_kept,
        len(deduped),
        len(result),
        float(result[0]["score"]) if result else 0.0,
    )
    return result


# ── Result conversion ─────────────────────────────────────────────────────────


def results_to_dicts(
    results_df,
    data_dir: Path,
    meta_by_scene: dict | None = None,
    fps: float = 24.0,
) -> list[dict]:
    """Convert a search result DataFrame to the template's card dicts."""
    out = []
    for row in results_df.to_dict("records"):
        d = {**row, "img_url": keyframe_url(str(row["filepath"]), data_dir)}
        if meta_by_scene is not None:
            meta = meta_by_scene.get(row.get("scene_id"))
            if meta:
                start_s = float(meta.get("start_time_s") or 0.0)
                d["timecode"] = to_smpte(start_s, fps) if start_s > 0 else ""
        out.append(d)
    return out


# ── Search orchestration ──────────────────────────────────────────────────────


def search_text(
    index: SearchIndex,
    query: str,
    tags: list[str],
    tag_index: dict,
    top_k: int,
    min_similarity: float = 0.0,
):
    """Run a text (optionally tag-filtered) semantic search."""
    from cinemateca.embeddings import SemanticSearch

    searcher = SemanticSearch(index.embeddings, index.kf_df, index.embedder)
    raw_k = top_k * 4
    if tags:
        df = searcher.combined(query, tags, tag_index, raw_k)
    else:
        df = searcher.by_text(query, raw_k)
    n_raw = len(df)
    top_raw = float(df["similarity"].iloc[0]) if n_raw and "similarity" in df.columns else 0.0
    if min_similarity > 0.0 and not df.empty and "similarity" in df.columns:
        df = df[df["similarity"] >= min_similarity].reset_index(drop=True)
    n_after_floor = len(df)
    if not df.empty and "scene_id" in df.columns:
        df = df.drop_duplicates(subset="scene_id", keep="first").reset_index(drop=True)
    df = df.head(top_k).reset_index(drop=True)
    logger.info(
        "search_text: query=%r top_k=%d tags=%s min_sim=%.3f "
        "raw_hits=%d top_score=%.3f kept_after_floor=%d dedup_kept=%d",
        query,
        top_k,
        tags or None,
        min_similarity,
        n_raw,
        top_raw,
        n_after_floor,
        len(df),
    )
    return df


def _mojica_search_defaults() -> dict:
    """Defaults the Mojica Buscar template needs for the initial empty state."""
    return {
        "query": "",
        "total": 0,
        "film_count": 0,
        "latency_ms": None,
        "active_mode": "text",
        "active_view": "grid",
        "selected_scene_id": None,
        "results": [],
        "films_by_id": {},
        "highlighted_tags": set(),
    }


def films_by_id_lookup(cfg: Any) -> dict:
    """Return ``{film.slug: film}`` for every registered film."""
    library_dir = Path(cfg.paths.library_dir)
    return {film.slug: film for film in list_films(library_dir)}


def build_search_context(ctx: FilmContext, cfg: Any | None = None) -> dict:
    """Build the per-film search-tab partial context."""
    tag_index = load_tag_index(ctx.metadata_dir)
    raw_tags = sorted(tag_index.keys()) if tag_index else []
    ctx_dict = _mojica_search_defaults()
    ctx_dict["available_tags"] = _filter_degenerate_tags(raw_tags)
    if cfg is not None:
        ctx_dict["films_by_id"] = films_by_id_lookup(cfg)
    return ctx_dict


def build_search_context_aggregate(cfg: Any) -> dict:
    """Build the aggregate search-tab context (union across all films)."""
    library_dir = Path(cfg.paths.library_dir)
    all_tags: set[str] = set()
    for film in list_films(library_dir):
        try:
            ctx = FilmContext.for_film(cfg, film.slug)
        except ValueError:
            continue
        tag_index = load_tag_index(ctx.metadata_dir)
        all_tags.update(tag_index.keys())
    ctx_dict = _mojica_search_defaults()
    ctx_dict["available_tags"] = _filter_degenerate_tags(sorted(all_tags))
    ctx_dict["films_by_id"] = films_by_id_lookup(cfg)
    return ctx_dict
