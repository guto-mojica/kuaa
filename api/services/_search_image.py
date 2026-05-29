"""Image-search functions — split from search.py (O-22).

Contains: upload validation and image-similarity search orchestration.
"""

from __future__ import annotations

from pathlib import Path

from api.services._search_text import SearchIndex
from cinemateca.search.types import UploadRejected  # re-exported; single source

# Server-side upload guards for image search. The cap is intentionally
# generous for a still frame (a 4K JPEG is well under this) while still
# refusing arbitrarily large / non-image payloads instead of streaming
# them into a tempfile and a CLIP forward pass.
MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # 8 MiB
ALLOWED_IMAGE_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
)


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


def search_image(index: SearchIndex, image_path: Path, top_k: int):
    """Run an image-similarity semantic search (sync; see search_text).

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
