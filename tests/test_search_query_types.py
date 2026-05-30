"""C3 — Query.text is a plain field; of_text is the text constructor."""

from __future__ import annotations

import pytest

from cinemateca.search.types import Query


def test_of_text_builds_text_query() -> None:
    q = Query.of_text("man on a horse")
    assert q.text == "man on a horse"
    assert q.image_path is None
    assert q.image_bytes is None


def test_text_field_is_readable_and_not_callable() -> None:
    q = Query.of_text("hi")
    # The field shadow is gone: q.text is the str value, not a bound method.
    assert isinstance(q.text, str)
    assert not callable(q.text)


def test_image_constructor_unchanged() -> None:
    from pathlib import Path

    q = Query.image(Path("/tmp/x.jpg"))
    assert q.image_path == Path("/tmp/x.jpg")
    assert q.text is None


def test_exactly_one_field_required() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        Query()
