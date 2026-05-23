"""Search service — CLIP semantic-search domain logic + index validation.

This module owns what used to live inline in ``api/routes/search.py``:

  * loading the on-disk CLIP index (``.npy`` embeddings + the
    ``index_mapping.json`` keyframe map) — previously a private
    ``_load_index`` cached with ``@lru_cache(maxsize=1)`` keyed only on
    the directory path string, so a *regenerated* index was never picked
    up without a process restart and a stale/corrupt index leaked
    between requests;
  * result-DataFrame → template-dict conversion (``_results_to_dicts``);
  * the text / image search orchestration the route did inline.

Two correctness additions Phase 3c makes on top of the pure extraction
(catalog.py / annotations.py were byte-preserving refactors; this one is
explicitly a validation phase per the plan):

  1. **mtime/size-aware cache invalidation.** The cache is keyed by the
     embeddings + mapping file paths AND their ``(st_mtime_ns, st_size)``
     stat signature. A regenerated index (different size/mtime) is
     re-loaded automatically — no restart, no manual ``cache_clear``.
     Acknowledged blind spot: an index regenerated to a byte-identical
     ``st_size`` AND an identical ``st_mtime_ns`` would not be detected;
     this is practically impossible on a real regeneration (the writer
     touches mtime and the content/size changes), and a content hash is
     the only complete fix — deliberately not done, out of scope for a
     single-worker dev server.

  2. **Index shape validation.** ``CLIPEmbedder.load`` performs NO
     row-count consistency check (see its docstring — it just
     ``np.load`` + ``json.load``). A mapping that declares fewer
     keyframes than the embeddings matrix has rows previously crashed
     ``/api/search`` with a pandas ``IndexError`` (the Phase-2
     ``xfail(strict=True)`` tripwire). Validation lives HERE, in the
     api/services layer, deliberately: it keeps the AI core's contract
     (``embeddings.py``) untouched and does not change embedding/model
     computation or artefact formats — it only refuses to *serve* an
     incoherent index, degrading to a clear "corrupt index" UI state.

Path resolution flows through :class:`FilmContext` (consistent with the
catalog / annotations services). The RAW merged tag index is reused from
``api/services/catalog.load_tag_index`` and passed verbatim to
``SemanticSearch.combined`` — that method self-normalizes (Phase 1c), so
pre-normalizing here would diverge from the characterized contract.

Thread-safety: the module-level cache is guarded by a simple
``threading.Lock`` covering the stat-check + (re)load + store. The lock
is held across the disk load; that is acceptable for the current
single-worker dev server. Request-level concurrency / a job runner is
Phase 4 — this module deliberately does not over-engineer past a
correctness-preserving lock.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from api.services.catalog import (
    derive_fps,
    keyframe_url,
    load_json,
    load_tag_index,
    to_smpte,
)
from api.services.film_context import FilmContext
from cinemateca.retrieval.hybrid import DEFAULT_RRF_K

if TYPE_CHECKING:
    import pandas as pd

    from cinemateca.retrieval.bm25 import BM25Index

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
    """True when ``tag`` looks like raw model output, not a curated label.

    Filters target the specific patterns Moondream leaks into
    ``scene_tags.json``:
      * empty / pure digit (``"1"``, ``"42"``)
      * longer than ``_DEGENERATE_TAG_MAX_LEN`` chars (full captions)
      * embedded ``.`` mid-string (sentence fragments)
      * trailing ``.`` paired with an article-led prefix
        (``"a-baby-in-a-basket."`` / ``"the-setting-is-a-farm."``).
        Bare trailing ``.`` survives (``"farm."``, ``"rural-field."``).
      * repeated-token sequences (``"gate-gate-gate"``)
      * digit-led enumeration prefix (``"1-cow"``, ``"3-sky"``)
      * de-dup numeric suffix (``"man-in-hat-2"``, ``"woman-in-dress-3"``)
      * >``_DEGENERATE_TAG_MAX_HYPHENS`` hyphens (curated tags are 0-2;
        more means multi-clause sentence fragments)

    Display-only — the underlying ``tag_index`` is unmodified, so a user
    who types a filtered tag into the URL still gets a working filter on
    the per-film search path.
    """
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


# Server-side upload guards for image search. The cap is intentionally
# generous for a still frame (a 4K JPEG is well under this) while still
# refusing arbitrarily large / non-image payloads instead of streaming
# them into a tempfile and a CLIP forward pass.
MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # 8 MiB
ALLOWED_IMAGE_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
)


class IndexStatus(str, Enum):
    """Outcome of attempting to load the search index."""

    OK = "ok"
    MISSING = "missing"  # .npy and/or mapping file absent
    CORRUPT = "corrupt"  # files present but shape-inconsistent / unreadable


@dataclass(frozen=True)
class SearchIndex:
    """A loaded (or failed) CLIP search index.

    ``status is IndexStatus.OK`` guarantees ``embeddings``, ``kf_df`` and
    ``embedder`` are populated and mutually shape-consistent. For
    ``MISSING`` / ``CORRUPT`` the route renders the no-index UI state
    instead of 500-ing; ``detail`` carries a log-friendly reason.
    """

    status: IndexStatus
    embeddings: object | None = None
    kf_df: object | None = None
    embedder: object | None = None
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status is IndexStatus.OK


class UploadRejected(Exception):
    """Raised by :func:`validate_upload` when an image upload fails the
    server-side size / content-type guards. The route turns this into a
    clear UI message rather than processing arbitrary input."""


# ── mtime/size-aware index cache ──────────────────────────────────────────────

# key: (slug_or_none, embeddings_path_str, mapping_path_str)
# value: (signature, SearchIndex) where signature is the combined stat tuple of
# both files at load time. A changed signature => reload.
# The slug component isolates each film's cache slot so per-film indices
# never collide (two films that happen to share a path string are distinct).
_index_cache: dict[tuple[str | None, str, str], tuple[tuple, SearchIndex]] = {}
_cache_lock = threading.Lock()


def _stat_sig(path: Path) -> tuple[int, int] | None:
    """Return ``(st_mtime_ns, st_size)`` for *path*, or ``None`` if absent.

    ``None`` participates in the cache signature so an index file
    appearing or disappearing also invalidates the cached entry.
    """
    try:
        st = path.stat()
    except (FileNotFoundError, NotADirectoryError):
        return None
    return (st.st_mtime_ns, st.st_size)


def _load_and_validate(emb_path: Path, map_path: Path) -> SearchIndex:
    """Load the index from disk and validate its shape coherence.

    Validation done HERE (not in the AI core ``embeddings.py``):

      * either file missing                       -> MISSING
      * unreadable / malformed mapping or npy     -> CORRUPT
      * embeddings row count != keyframe-map rows  -> CORRUPT

    The successful path is byte-equivalent to the old route's
    ``CLIPEmbedder.load`` + ``CLIPEmbedder()`` construction, so a
    well-formed index behaves exactly as before.
    """
    if not emb_path.exists() or not map_path.exists():
        logger.warning("Search index not found at %s", emb_path.parent)
        return SearchIndex(IndexStatus.MISSING, detail="index files absent")

    from cinemateca.models.clip.openclip import OpenClipEmbedder

    try:
        embeddings, mapping, kf_df = OpenClipEmbedder.load(emb_path, map_path)
    except Exception as exc:  # malformed .npy / .json, missing keys, etc.
        logger.warning("Search index unreadable (%s): %s", emb_path.parent, exc)
        return SearchIndex(IndexStatus.CORRUPT, detail=f"unreadable: {exc}")

    n_emb = int(getattr(embeddings, "shape", [0])[0])
    n_map = len(kf_df)
    declared = mapping.get("total_vectors")
    if n_emb != n_map:
        logger.warning(
            "Corrupt search index: %d embedding rows vs %d keyframe-map " "rows (%s)",
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
            "Corrupt search index: mapping declares total_vectors=%s but "
            "has %d keyframe rows (%s)",
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
    """Return the (cached) :class:`SearchIndex` for *ctx*'s embeddings dir.

    The result is cached keyed by the embeddings + mapping file paths and
    their ``(mtime_ns, size)`` signature. If either file is changed,
    added or removed since the cached load, the index is reloaded
    transparently — a regenerated index is picked up WITHOUT a process
    restart (the prior ``@lru_cache`` keyed only on the dir string never
    did this).

    Always returns a ``SearchIndex`` (never ``None`` / never raises for a
    missing/corrupt index): the route inspects ``.status`` and renders
    the no-index UI state for ``MISSING`` / ``CORRUPT``.
    """
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
    """Drop every cached index entry (test-isolation hook).

    Production code never needs this — the stat signature handles
    invalidation. ``tests/conftest.py`` calls it per test so a
    populated/corrupt index from one test cannot leak into the next
    (the role the old ``search._load_index.cache_clear()`` played).
    """
    with _cache_lock:
        _index_cache.clear()


# ── BM25 loader: disk reads + merged tag-index + mtime+size cache ────────────


def _file_stamp(path: Path) -> tuple[float, int]:
    """``(mtime, size)`` of a file, or ``(0.0, 0)`` if absent.

    Used as a cache key — bumps on any write, including ones that land
    within the same mtime tick (size differs).
    """
    try:
        st = path.stat()
    except FileNotFoundError:
        return (0.0, 0)
    return (st.st_mtime, st.st_size)


@lru_cache(maxsize=32)
def _cached_bm25_index(
    metadata_dir: str,
    descriptions_stamp: tuple[float, int],
    scene_tags_stamp: tuple[float, int],
    manual_annotations_stamp: tuple[float, int],
    stopwords_lang: str | None,
    k1: float,
    b: float,
) -> BM25Index:
    """Build a BM25 index for the given (already-stamped) metadata dir.

    The three stamp tuples form the cache key: any write to any of the
    three source files bumps either mtime or size, forcing a rebuild.
    The path string is in the key too so two films don't collide.

    Source files (all under ``metadata_dir``):
      * ``scene_descriptions.json`` — Moondream output (list of dicts).
      * ``scene_tags.json`` — LLM-tag output (INT scene_id keys).
      * ``manual_annotations.json`` — manual tags (STR scene_id keys).

    Tag merge semantics come from ``catalog.load_tag_index`` (single
    source of truth for scene_id normalisation across both tag files).
    """
    import json as _json
    from pathlib import Path as _Path

    from cinemateca.retrieval.bm25 import BM25Index as _BM25Index

    md = _Path(metadata_dir)
    descriptions_path = md / "scene_descriptions.json"
    descriptions: list[dict] = []
    if descriptions_path.exists():
        try:
            data = _json.loads(descriptions_path.read_text())
            descriptions = data if isinstance(data, list) else []
        except _json.JSONDecodeError:
            logger.warning("BM25: malformed %s; using empty descriptions", descriptions_path)

    # Merged LLM ⊕ manual tag index via the existing catalog helper —
    # the single source of truth for the merge semantics (it owns
    # scene_id normalisation).
    tag_index = load_tag_index(md) or {}

    return _BM25Index.build(
        descriptions=descriptions,
        tag_index=tag_index,
        stopwords_lang=stopwords_lang,
        k1=k1,
        b=b,
    )


def _get_bm25_index_for_ctx(ctx: FilmContext) -> BM25Index:
    """Load + cache the BM25 index for one film.

    Cache invalidates when any of three source files changes:
      * ``scene_descriptions.json`` (Moondream output)
      * ``scene_tags.json`` (LLM-tag output)
      * ``manual_annotations.json`` (manual tags)

    The cache holds the 32 most-recently-used films (more than enough
    for any plausible library size). The ``get_config`` import is
    deferred to keep this module loadable without the FastAPI app
    config wired up (matters for tests that import the service module
    in isolation).
    """
    from api.deps import get_config

    cfg = get_config()
    bm25_cfg = getattr(cfg.search, "bm25", None)
    stopwords_lang = getattr(bm25_cfg, "stopwords_lang", None) if bm25_cfg else None
    k1 = float(getattr(bm25_cfg, "k1", 1.5)) if bm25_cfg else 1.5
    b = float(getattr(bm25_cfg, "b", 0.75)) if bm25_cfg else 0.75

    md = ctx.metadata_dir
    return _cached_bm25_index(
        str(md),
        _file_stamp(md / "scene_descriptions.json"),
        _file_stamp(md / "scene_tags.json"),
        _file_stamp(md / "manual_annotations.json"),
        stopwords_lang,
        k1,
        b,
    )


# ── Per-film helpers + aggregate search ───────────────────────────────────────

# Canonical filenames for the per-film CLIP index.  These mirror the
# ``config/default.yaml`` → ``embeddings.*`` values and are used as
# defaults when ``cfg.embeddings`` is not present (e.g. in unit tests
# that supply a minimal SimpleNamespace config).
_DEFAULT_EMBEDDINGS_FILENAME = "keyframe_embeddings.npy"
_DEFAULT_MAPPING_FILENAME = "index_mapping.json"


def _get_embedder(cfg: Any) -> object:
    """Return a fresh ``OpenClipEmbedder`` instance.

    Extracted to module scope so unit tests can monkeypatch
    ``api.services.search._get_embedder`` to avoid loading the real CLIP
    model. The ``cfg`` argument is accepted for API consistency and future
    use (e.g. routing to a different backend via cfg) but is currently
    ignored.
    """
    from cinemateca.models.clip.openclip import OpenClipEmbedder

    return OpenClipEmbedder()


def _get_search_index(cfg: Any, slug: str) -> SearchIndex:
    """Return the (cached) :class:`SearchIndex` for the film identified by *slug*.

    Resolves the per-film embeddings directory via
    :meth:`FilmContext.for_film`, then delegates to :func:`load_index`
    with the canonical filenames (read from ``cfg.embeddings`` when
    available, falling back to the constants above for test configs that
    only supply ``paths.library_dir``).
    """
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
    """``True`` iff at least one registered film has an OK :class:`SearchIndex`.

    Lets the route distinguish two empty-hit cases:

      * library has no indexed films yet → render "No search index found"
        (user needs to run the embeddings pipeline);
      * library has indexed films but the query produced zero hits above
        ``min_similarity`` → render "No results" (the query simply didn't
        match anything in the corpus — a normal outcome, not a setup error).
    """
    from cinemateca.library import scan_library

    library_dir = Path(cfg.paths.library_dir)
    for film in scan_library(library_dir):
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
    retriever_mode: str = "clip",
    sem_w: float = 0.70,
    bm25_w: float = 0.30,
) -> list[dict]:
    """Run per-film search and merge top results by score.

    For ``modality == "text"`` only in this plan; image / audio / fusion
    modalities are wired in Plans 3-5. The merger is a plain sort by
    cosine score across films — comparable because every film uses the
    same CLIP backbone (registry default).

    When ``tags`` is non-empty, each film's hits are restricted to scenes
    whose ``scene_id`` is in EVERY selected tag's list (AND intersection),
    mirroring ``SemanticSearch.combined``'s per-film semantics. A film
    that lacks any of the selected tags contributes zero hits.

    ``min_similarity`` is a cosine-score floor applied per hit before the
    cross-film merge. CLIP returns top-K nearest neighbours unconditionally,
    so unrelated queries used to surface 8 noise scenes — the threshold
    drops anything below it. A query whose top result is under the floor
    returns ``[]`` (the route renders the no-index UI state, which the
    template now also covers for "no results above threshold").

    ``retriever_mode`` / ``sem_w`` / ``bm25_w`` are D1 pre-staging: the
    route's signature now passes them through, but the per-film fan-out
    here still runs pure-CLIP regardless. Task D2 (next in the plan)
    fills in the actual cross-film hybrid dispatch — keeping the kwargs
    here with legacy-clip defaults lets D1 land as a single coherent
    commit (route + service signature) without forcing the dispatcher
    to ship half-built.
    """
    from cinemateca.library import scan_library
    from cinemateca.scene_ids import normalize_tag_index, scene_id_key

    if modality != "text":
        raise NotImplementedError(
            f"modality={modality!r} lands in a later plan; only 'text' is supported here"
        )

    library_dir = Path(cfg.paths.library_dir)
    # Materialise the film list BEFORE loading the embedder.  When the library
    # is empty (0 registered films) we return early and avoid the ~4 s CLIP
    # model initialisation that _get_embedder triggers.
    films = list(scan_library(library_dir))
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
            # Registered film whose directory has been removed manually —
            # skip silently rather than crash the whole aggregate.
            logger.warning("aggregate_search: skip film %s — %s", film.slug, exc)
            continue
        if idx.status is not IndexStatus.OK:
            logger.info(
                "aggregate_search: skip film %s — index status %s",
                film.slug,
                idx.status,
            )
            continue
        # Load the film's keyframe metadata ONCE per film so timecode lookup
        # is O(1) per hit. ``meta_by_scene`` may be empty (unprocessed film
        # or missing JSON) — in that case timecode falls back to "" and the
        # template hides the span.
        ctx = FilmContext.for_film(cfg, film.slug)
        kf_meta = load_json(ctx.metadata_dir / "keyframes_metadata.json") or []
        fps = derive_fps(kf_meta)
        meta_by_scene = {e["scene_id"]: e for e in kf_meta if "scene_id" in e}

        # Tag pre-filter (AND intersection across selected tags). Mirrors
        # SemanticSearch.combined: normalize the raw tag_index to canonical
        # str scene ids, intersect membership sets, skip the film entirely
        # if any selected tag is missing or the intersection is empty.
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
        # Per-film diagnostic: the score distribution is the input the
        # threshold/reranker operates on, so log enough to tune both.
        # n_vectors is the index size; top3 lets you see whether the
        # query genuinely has signal in this film vs being uniform noise.
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
    # Dedupe by (film_slug, scene_id). With multiple keyframes per scene
    # (Phase-1 density fix) the same scene can rank N times in a row.
    # Keeping the first occurrence per scene preserves the highest-score
    # keyframe (the list is sorted descending). Result: at most one card
    # per scene in the UI, and the displayed keyframe is the one that
    # actually matches the query — not always the middle frame.
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
    """Convert a search result DataFrame to the template's card dicts.

    When ``meta_by_scene`` is supplied (a ``{scene_id: kf_entry}`` dict
    from ``keyframes_metadata.json``), each result row is enriched with a
    SMPTE ``timecode`` field computed from ``start_time_s``. Without it
    the behaviour is byte-equivalent to the prior route implementation.
    """
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


# ── Upload validation ─────────────────────────────────────────────────────────


def validate_upload(filename: str | None, content_type: str | None, data: bytes) -> str:
    """Validate an image-search upload; return a safe file suffix.

    Rejects (raising :class:`UploadRejected`) when:

      * the body is empty,
      * the body exceeds :data:`MAX_UPLOAD_BYTES`,
      * the declared content-type is present and is not ``image/*``,
      * the filename suffix is not a known image suffix.

    Returns the lower-cased suffix to use for the temp file (defaulting
    to ``.jpg`` only when a content-type positively identifies an image
    but the filename had no usable extension).
    """
    if not data:
        raise UploadRejected("empty upload")
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadRejected(f"file too large ({len(data)} bytes > {MAX_UPLOAD_BYTES} limit)")

    ctype = (content_type or "").split(";", 1)[0].strip().lower()
    if ctype and not ctype.startswith("image/"):
        raise UploadRejected(f"unsupported content-type: {ctype!r}")

    suffix = Path(filename or "").suffix.lower()
    if suffix:
        if suffix not in ALLOWED_IMAGE_SUFFIXES:
            raise UploadRejected(f"unsupported file type: {suffix!r}")
        return suffix
    # No suffix on the filename: only accept if the content-type itself
    # asserted an image (ctype.startswith("image/") already checked).
    if ctype.startswith("image/"):
        return ".jpg"
    raise UploadRejected("missing image file extension and content-type")


# ── Search orchestration ──────────────────────────────────────────────────────


def search_text(
    index: SearchIndex,
    query: str,
    tags: list[str],
    tag_index: dict,
    top_k: int,
    min_similarity: float = 0.0,
):
    """Run a text (optionally tag-filtered) semantic search.

    Mirrors the prior route logic exactly: with ``tags`` it calls
    ``SemanticSearch.combined`` passing the RAW merged ``tag_index``
    (``combined`` self-normalizes — Phase 1c contract preserved); without
    tags it calls ``by_text``. Caller (route) is responsible for running
    this in an executor — kept sync here so the service stays
    framework-agnostic and unit-testable without an event loop.

    ``min_similarity`` post-filters the result DataFrame (CLIP returns
    top-K unconditionally, so unrelated queries surface noise scenes;
    the threshold drops anything below the cosine floor). 0.0 disables
    the filter (default for back-compat with unit tests).
    """
    from cinemateca.embeddings import SemanticSearch

    searcher = SemanticSearch(index.embeddings, index.kf_df, index.embedder)
    # The underlying searcher returns the global top-K by similarity; with
    # multiple keyframes per scene that top-K may concentrate inside one
    # scene's keyframe block, starving other scenes. Ask for a wider
    # window (top_k * kf_per_scene) so the post-dedupe top-K still has
    # ``top_k`` distinct scenes to choose from. The wider window only
    # affects ranking, not embedding cost.
    raw_k = top_k * 4  # 4× is the configured ceiling for keyframes_per_scene
    if tags:
        df = searcher.combined(query, tags, tag_index, raw_k)
    else:
        df = searcher.by_text(query, raw_k)
    n_raw = len(df)
    top_raw = float(df["similarity"].iloc[0]) if n_raw and "similarity" in df.columns else 0.0
    if min_similarity > 0.0 and not df.empty and "similarity" in df.columns:
        df = df[df["similarity"] >= min_similarity].reset_index(drop=True)
    # Dedupe by scene_id (Phase-1 density fix). The DataFrame is already
    # ordered by similarity descending, so ``drop_duplicates`` keeps the
    # first occurrence per scene = the best-matching keyframe of that
    # scene. Trim to ``top_k`` AFTER dedup so the UI gets the requested
    # number of *scenes*, not keyframes.
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


def search_hybrid(
    index: SearchIndex,
    *,
    bm25: BM25Index | None,
    query: str,
    tags: list[str],
    tag_index: dict,
    top_k: int,
    min_similarity: float,
    retriever_mode: str,
    sem_w: float,
    bm25_w: float,
    rrf_k: int = DEFAULT_RRF_K,
) -> pd.DataFrame:
    """Dispatch text search across one of three retrieval pipelines.

    Returns the same DataFrame shape ``search_text`` returns (columns:
    ``scene_id``, ``similarity``, plus the keyframe columns). The route
    layer cannot tell which mode produced the result — that's the
    whole point of the dispatcher.

    Args:
        index: per-film CLIP search index.
        bm25: per-film BM25 index, or ``None`` when ``retriever_mode`` is
            ``"clip"`` and BM25 wasn't loaded (cheap skip).
        retriever_mode: one of ``"clip" | "bm25" | "hybrid"``.
        sem_w, bm25_w: only consulted in ``"hybrid"`` mode.
        min_similarity: cosine floor for the CLIP path. Applied only on
            ``"clip"`` mode and on the CLIP side of ``"hybrid"`` (where
            it pre-filters before RRF). Not applied on ``"bm25"`` mode —
            BM25 scores are not in the cosine-similarity scale.

    Mode behaviour:
        * ``"clip"`` — delegates to :func:`search_text` unchanged.
          Regression pin for pre-M2 ordering.
        * ``"bm25"`` — runs BM25 only, formats the result the same way
          ``search_text`` would (mapping back to the index's metadata).
        * ``"hybrid"`` — runs both, fuses by weighted RRF, returns
          ``top_k`` by fused-score.

    Graceful fallback: when ``bm25`` is ``None`` or its underlying
    ``model`` is ``None`` (empty corpus), any non-``"clip"`` mode
    transparently degrades to CLIP-only — the UI degrades without
    raising, the route still serves a result.
    """
    if retriever_mode == "clip":
        return search_text(index, query, tags, tag_index, top_k, min_similarity)

    if bm25 is None or bm25.model is None:
        logger.info(
            "search_hybrid: bm25 index empty/None; falling back to clip (mode=%s requested)",
            retriever_mode,
        )
        return search_text(index, query, tags, tag_index, top_k, min_similarity)

    # Mirrors search_text's keyframe-density widening (4× ceiling).
    raw_k = top_k * 4

    if retriever_mode == "bm25":
        # BM25 scores aren't cosine — min_similarity does not apply.
        hits = bm25.query(query, top_k=raw_k)
        return _bm25_hits_to_dataframe(hits, index, tags, tag_index, top_k)

    # retriever_mode == "hybrid". min_similarity flows through search_text
    # below, pre-filtering the CLIP side before RRF fusion.
    clip_df = search_text(index, query, tags, tag_index, raw_k, min_similarity)
    clip_ranked: list[tuple[int, float]] = (
        [(int(row.scene_id), float(row.similarity)) for row in clip_df.itertuples(index=False)]
        if not clip_df.empty
        else []
    )
    bm25_hits = bm25.query(query, top_k=raw_k)

    from cinemateca.retrieval.hybrid import fuse_rrf

    fused = fuse_rrf(clip_ranked, bm25_hits, sem_w=sem_w, bm25_w=bm25_w, k_rrf=rrf_k)[:top_k]

    return _fused_to_dataframe(fused, clip_df, index, tags, tag_index, top_k)


def _bm25_hits_to_dataframe(
    hits: list[tuple[int, float]],
    index: SearchIndex,
    tags: list[str],
    tag_index: dict,
    top_k: int,
) -> pd.DataFrame:
    """Materialise BM25-only hits into the ``search_text`` DataFrame shape.

    BM25 scores are not in CLIP's cosine-similarity scale, but the
    template only cares about ordering. We expose the BM25 score as
    ``similarity`` for shape-compat; routes that surface raw scores can
    distinguish via the ``retriever`` query param.
    """
    import pandas as pd

    if not hits:
        return pd.DataFrame(columns=["scene_id", "similarity"])
    df = pd.DataFrame(hits, columns=["scene_id", "similarity"])
    if tags and tag_index:
        keep_sids: set[int] = set()
        for t in tags:
            for sid in tag_index.get(t, []):
                keep_sids.add(int(sid))
        df = df[df["scene_id"].isin(keep_sids)].reset_index(drop=True)
    if hasattr(index, "kf_df") and index.kf_df is not None:
        df = df.merge(
            index.kf_df.drop_duplicates(subset="scene_id", keep="first"),
            on="scene_id",
            how="left",
        )
    return df.head(top_k).reset_index(drop=True)


def _fused_to_dataframe(
    fused: list[tuple[int, float]],
    clip_df: pd.DataFrame,
    index: SearchIndex,
    tags: list[str],
    tag_index: dict,
    top_k: int,
) -> pd.DataFrame:
    """Materialise the fused ranking, reusing ``clip_df`` rows when present.

    BM25-only hits (scenes the CLIP top-K didn't surface) are back-filled
    from ``index.kf_df`` so every row carries the keyframe columns the
    template expects.
    """
    import pandas as pd

    if not fused:
        return pd.DataFrame(columns=["scene_id", "similarity"])
    fused_df = pd.DataFrame(fused, columns=["scene_id", "fused_score"])
    if not clip_df.empty:
        merged = fused_df.merge(
            clip_df.drop(columns=["similarity"], errors="ignore"),
            on="scene_id",
            how="left",
        )
    else:
        merged = fused_df
    merged["similarity"] = merged["fused_score"]
    merged = merged.drop(columns=["fused_score"])
    if tags and tag_index:
        keep_sids: set[int] = set()
        for t in tags:
            for sid in tag_index.get(t, []):
                keep_sids.add(int(sid))
        merged = merged[merged["scene_id"].isin(keep_sids)].reset_index(drop=True)
    if hasattr(index, "kf_df") and index.kf_df is not None:
        # Backfill keyframe metadata for BM25-only hits.
        if "img_filename" in merged.columns:
            missing_mask = merged["img_filename"].isna()
        else:
            missing_mask = pd.Series([False] * len(merged))
        missing = merged[missing_mask]
        if not missing.empty:
            patched = missing[["scene_id", "similarity"]].merge(
                index.kf_df.drop_duplicates(subset="scene_id", keep="first"),
                on="scene_id",
                how="left",
            )
            merged = pd.concat(
                [merged[~merged["scene_id"].isin(patched["scene_id"])], patched],
                ignore_index=True,
            )
            merged = merged.sort_values("similarity", ascending=False).reset_index(drop=True)
    return merged.head(top_k).reset_index(drop=True)


def search_image(index: SearchIndex, image_path: Path, top_k: int):
    """Run an image-similarity semantic search (sync; see :func:`search_text`).

    Applies the same scene_id dedupe as :func:`search_text` so the UI
    receives at most one card per scene, displaying the best-matching
    keyframe (rather than three near-duplicate rows from the same shot).
    """
    from cinemateca.embeddings import SemanticSearch

    searcher = SemanticSearch(index.embeddings, index.kf_df, index.embedder)
    df = searcher.by_image(image_path, top_k * 4)
    if not df.empty and "scene_id" in df.columns:
        df = df.drop_duplicates(subset="scene_id", keep="first").reset_index(drop=True)
    return df.head(top_k).reset_index(drop=True)


def _mojica_search_defaults() -> dict:
    """Defaults the Mojica Buscar template (``partials/search.html``)
    needs whenever no actual query has been issued.

    Task 10 introduces a richer template context — query state, view
    toggle, results list, film lookup, highlighted tags — that previous
    tab-renders did not surface. These defaults let the page render the
    initial "type a query to search" empty state with no special casing
    on the template side.

    The per-modality result list is intentionally empty here. Task 11
    fills it with ``.b-card``-shaped dicts produced by the /api/search
    handlers; ``films_by_id`` is populated lazily by callers that have a
    cfg in hand (see :func:`films_by_id_lookup`).
    """
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
    """Return ``{film.slug: film}`` for every registered film.

    Task 11's ``.b-card`` markup looks up ``films_by_id[r.film_slug]`` to
    pull the film title + year onto each result card; the lookup is built
    here so both the per-film and aggregate routes (and the
    ``build_search_context*`` builders) populate the same shape.

    Returns an empty dict when the library directory is absent —
    consistent with :func:`cinemateca.library.scan_library`'s contract.
    Templates should treat the dict as a best-effort lookup
    (``films_by_id.get(slug)``); cards whose ``film_slug`` is missing
    still render with sensible fallbacks.
    """
    from cinemateca.library import scan_library

    library_dir = Path(cfg.paths.library_dir)
    return {film.slug: film for film in scan_library(library_dir)}


def build_search_context(ctx: FilmContext, cfg: Any | None = None) -> dict:
    """Build the per-film search-tab partial context.

    Uses the RAW merged tag index (only its keys feed ``available_tags``
    — identical to the normalized index's keys) and runs them through
    ``_filter_degenerate_tags`` so the pill grid stays clean even when
    ``scene_tags.json`` carries leaked caption fragments.

    Mojica-redesign keys (Task 10) live alongside ``available_tags`` so
    the rewritten template can render the empty state without forcing
    every route to populate them. The ``query`` / ``total`` / ``results``
    defaults are overwritten by ``/api/search`` responses once a query
    fires.

    ``cfg`` is optional for back-compat: when supplied, ``films_by_id``
    is populated via :func:`films_by_id_lookup` so Task-11's ``.b-card``
    template can resolve film titles/years on hits returned by the same
    request. When omitted (legacy callers), ``films_by_id`` stays empty
    and the template falls back to safe-get behaviour.
    """
    tag_index = load_tag_index(ctx.metadata_dir)
    raw_tags = sorted(tag_index.keys()) if tag_index else []
    ctx_dict = _mojica_search_defaults()
    ctx_dict["available_tags"] = _filter_degenerate_tags(raw_tags)
    if cfg is not None:
        ctx_dict["films_by_id"] = films_by_id_lookup(cfg)
    return ctx_dict


def build_search_context_aggregate(cfg: Any) -> dict:
    """Build the aggregate search-tab context (union across all films).

    Mirrors :func:`api.services.catalog.build_scenes_context_aggregate`'s
    tag-union pattern: walks the library registry, unions every film's
    tag-index keys, filters degenerate entries, and returns the same
    ``available_tags`` key the per-film builder exposes — so the
    ``partials/search.html`` template renders identically in either mode.

    Mojica-redesign keys (Task 10) are merged in via
    :func:`_mojica_search_defaults` so the aggregate path and per-film
    path expose the same context shape. ``films_by_id`` is populated
    here so the template's title/year lookup resolves on every card.
    """
    from cinemateca.library import scan_library

    library_dir = Path(cfg.paths.library_dir)
    all_tags: set[str] = set()
    for film in scan_library(library_dir):
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
