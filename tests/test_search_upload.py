"""Unit tests for cinemateca.search.upload — image upload validation."""

from __future__ import annotations

import pytest

from cinemateca.search.types import UploadRejected
from cinemateca.search.upload import (
    ALLOWED_IMAGE_SUFFIXES,
    MAX_UPLOAD_BYTES,
    validate_upload,
)


def test_max_bytes_is_8_mib():
    assert MAX_UPLOAD_BYTES == 8 * 1024 * 1024


def test_allowed_suffixes_include_jpeg():
    assert ".jpg" in ALLOWED_IMAGE_SUFFIXES
    assert ".jpeg" in ALLOWED_IMAGE_SUFFIXES
    assert ".png" in ALLOWED_IMAGE_SUFFIXES


def test_rejects_empty():
    with pytest.raises(UploadRejected, match="empty upload"):
        validate_upload("frame.jpg", "image/jpeg", b"")


def test_rejects_oversize():
    data = b"x" * (MAX_UPLOAD_BYTES + 1)
    with pytest.raises(UploadRejected, match="too large"):
        validate_upload("frame.jpg", "image/jpeg", data)


def test_rejects_non_image_ctype():
    with pytest.raises(UploadRejected, match="content-type"):
        validate_upload("frame.bin", "application/octet-stream", b"\x00")


def test_rejects_unsupported_suffix():
    with pytest.raises(UploadRejected, match="unsupported file type"):
        validate_upload("frame.exe", "image/jpeg", b"\xff\xd8")


def test_accepts_jpeg():
    assert validate_upload("frame.jpg", "image/jpeg", b"\xff\xd8\xff\xd9") == ".jpg"


def test_accepts_png():
    assert validate_upload("frame.png", "image/png", b"\x89PNG") == ".png"


def test_accepts_ctype_with_charset():
    assert validate_upload("frame.jpg", "image/jpeg; charset=binary", b"\xff\xd8") == ".jpg"


def test_no_suffix_with_image_ctype_defaults_to_jpg():
    assert validate_upload("frame", "image/jpeg", b"\xff\xd8") == ".jpg"


def test_no_suffix_no_image_ctype_rejects():
    with pytest.raises(UploadRejected, match="missing image file extension"):
        validate_upload(None, None, b"\xff\xd8")
