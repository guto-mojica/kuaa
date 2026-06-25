"""Degenerate-tag filter — drops raw model-output fragments from the
displayed tag vocabulary.

Display-only: the underlying tag_index is unmodified, so a request that
arrives with a degenerate-looking ``tags=...`` query still works.
"""

from __future__ import annotations

import re

_MAX_LEN = 40
_MAX_HYPHENS = 2
_REPEATED_TOKEN_RE = re.compile(r"\b(\w+)(?:[-\s]\1\b){2,}", re.IGNORECASE)
_TRAILING_NUMBER_RE = re.compile(r"-\d+$")
_DIGIT_LED_RE = re.compile(r"^\d+")
_ARTICLE_LED_RE = re.compile(r"^(a|the)-", re.IGNORECASE)


def is_degenerate_tag(tag: str) -> bool:
    """True when ``tag`` looks like raw model output, not a curated label.

    The filter targets the specific patterns Moondream leaks into
    ``scene_tags.json``: long captions, repeated tokens, enumerated
    lists, numeric dedup suffixes.
    """
    if not tag:
        return True
    if tag.isdigit():
        return True
    if len(tag) > _MAX_LEN:
        return True
    if "." in tag.rstrip("."):
        return True
    if tag.endswith(".") and _ARTICLE_LED_RE.match(tag):
        return True
    if _REPEATED_TOKEN_RE.search(tag):
        return True
    if tag.count("-") > _MAX_HYPHENS:
        return True
    if _DIGIT_LED_RE.match(tag):
        return True
    if _TRAILING_NUMBER_RE.search(tag):
        return True
    return False


def filter_degenerate_tags(tags) -> list[str]:
    """Drop degenerate-looking tag strings from the displayed vocabulary."""
    return [t for t in tags if not is_degenerate_tag(t)]
