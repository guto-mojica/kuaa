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

from api.services.catalog import keyframe_url, load_tag_index
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

# key: (embeddings_path_str, mapping_path_str)
# value: (signature, SearchIndex) where signature is the stat tuple of
# both files at load time. A changed signature => reload.
_index_cache: dict[tuple[str, str], tuple[tuple, SearchIndex]] = {}
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

    from cinemateca.embeddings import CLIPEmbedder

    try:
        embeddings, mapping, kf_df = CLIPEmbedder.load(emb_path, map_path)
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

    embedder = CLIPEmbedder()
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
    key = (str(emb_path), str(map_path))
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


# ── Result conversion ─────────────────────────────────────────────────────────

def results_to_dicts(results_df, data_dir: Path) -> list[dict]:
    """Convert a search result DataFrame to the template's card dicts.

    Byte-equivalent to the prior route ``_results_to_dicts``: each row
    gains an ``img_url`` resolved via the shared catalog primitive.
    """
    return [
        {**row, "img_url": keyframe_url(str(row["filepath"]), data_dir)}
        for row in results_df.to_dict("records")
    ]


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
