"""Image upload validation for image-similarity search.

Server-side guards for image-search uploads. The cap is intentionally
generous for a still frame (a 4K JPEG is well under this) while still
refusing arbitrarily large / non-image payloads instead of streaming
them into a tempfile and a CLIP forward pass.
"""

from __future__ import annotations

from pathlib import Path

from cinemateca.search.types import UploadRejected

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
