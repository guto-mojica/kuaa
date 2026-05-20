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
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from api.services.catalog import keyframe_url, load_tag_index, to_smpte
from api.services.film_context import FilmContext

logger = logging.getLogger(__name__)

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
            "Corrupt search index: %d embedding rows vs %d keyframe-map "
            "rows (%s)",
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
            detail=(
                f"mapping total_vectors={declared} != {n_map} keyframe rows"
            ),
        )

    embedder = OpenClipEmbedder()
    logger.info("Search index loaded: %d vectors", n_map)
    return SearchIndex(
        IndexStatus.OK, embeddings=embeddings, kf_df=kf_df, embedder=embedder
    )


def load_index(ctx: FilmContext, *, mapping_filename: str,
                embeddings_filename: str) -> SearchIndex:
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


def aggregate_search(
    cfg: Any,
    *,
    query: str,
    modality: str,
    top_k: int,
) -> list[dict]:
    """Run per-film search and merge top results by score.

    For ``modality == "text"`` only in this plan; image / audio / fusion
    modalities are wired in Plans 3-5. The merger is a plain sort by
    cosine score across films — comparable because every film uses the
    same CLIP backbone (registry default).
    """
    from cinemateca.library import scan_library

    if modality != "text":
        raise NotImplementedError(
            f"modality={modality!r} lands in a later plan; only 'text' is supported here"
        )

    library_dir = Path(cfg.paths.library_dir)
    embedder = _get_embedder(cfg)

    text_vec: np.ndarray = embedder.encode_text(query)  # type: ignore[union-attr]
    norm = float(np.linalg.norm(text_vec))
    text_vec = text_vec / (norm + 1e-12)

    all_hits: list[dict] = []
    for film in scan_library(library_dir):
        try:
            idx = _get_search_index(cfg, film.slug)
        except ValueError as exc:
            # Registered film whose directory has been removed manually —
            # skip silently rather than crash the whole aggregate.
            logger.warning(
                "aggregate_search: skip film %s — %s", film.slug, exc
            )
            continue
        if idx.status is not IndexStatus.OK:
            logger.info(
                "aggregate_search: skip film %s — index status %s",
                film.slug,
                idx.status,
            )
            continue
        scores: np.ndarray = idx.embeddings @ text_vec  # type: ignore[operator]
        for i, score in enumerate(scores):
            row = idx.kf_df.iloc[i]  # type: ignore[union-attr]
            all_hits.append(
                {
                    "film_slug": film.slug,
                    "film_title": film.title,
                    "scene_id": int(row["scene_id"]),
                    "score": float(score),
                    "keyframe_path": str(row["filepath"]),
                }
            )

    all_hits.sort(key=lambda h: -h["score"])
    return all_hits[:top_k]


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

def validate_upload(filename: str | None, content_type: str | None,
                     data: bytes) -> str:
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
        raise UploadRejected(
            f"file too large ({len(data)} bytes > {MAX_UPLOAD_BYTES} limit)"
        )

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

def search_text(index: SearchIndex, query: str, tags: list[str],
                 tag_index: dict, top_k: int):
    """Run a text (optionally tag-filtered) semantic search.

    Mirrors the prior route logic exactly: with ``tags`` it calls
    ``SemanticSearch.combined`` passing the RAW merged ``tag_index``
    (``combined`` self-normalizes — Phase 1c contract preserved); without
    tags it calls ``by_text``. Caller (route) is responsible for running
    this in an executor — kept sync here so the service stays
    framework-agnostic and unit-testable without an event loop.
    """
    from cinemateca.embeddings import SemanticSearch

    searcher = SemanticSearch(index.embeddings, index.kf_df, index.embedder)
    if tags:
        return searcher.combined(query, tags, tag_index, top_k)
    return searcher.by_text(query, top_k)


def search_image(index: SearchIndex, image_path: Path, top_k: int):
    """Run an image-similarity semantic search (sync; see :func:`search_text`)."""
    from cinemateca.embeddings import SemanticSearch

    searcher = SemanticSearch(index.embeddings, index.kf_df, index.embedder)
    return searcher.by_image(image_path, top_k)


def build_search_context(ctx: FilmContext) -> dict:
    """Build the search-tab partial context (the ``available_tags`` list).

    Moved verbatim from ``api/routes/search.build_search_context`` so the
    ``/tab/search`` fragment and the ``/search`` full page keep rendering
    identical markup. Uses the RAW merged tag index (only its keys feed
    ``available_tags`` — identical to the normalized index's keys).
    """
    tag_index = load_tag_index(ctx.metadata_dir)
    available_tags = sorted(tag_index.keys()) if tag_index else []
    return {"available_tags": available_tags}
