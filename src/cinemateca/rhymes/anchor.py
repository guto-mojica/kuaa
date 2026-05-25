"""Anchor URL parameter parsing — '<slug>:<scene_id>' tuple form."""
from __future__ import annotations

from typing import Any


def parse_anchor(anchor: str | None) -> tuple[str | None, int | None]:
    """Split the ``?anchor=`` query value into ``(slug, scene_id)``.

    Accepts ``"<slug>/<scene_id>"``; anything else (missing param,
    missing slash, non-int scene_id) returns ``(None, None)`` so the
    caller falls back to the default-anchor branch.
    """
    if not anchor or "/" not in anchor:
        return None, None
    slug, scene_id_s = anchor.split("/", 1)
    if not slug:
        return None, None
    try:
        return slug, int(scene_id_s)
    except (TypeError, ValueError):
        return None, None


def default_anchor(films: list[Any]) -> tuple[str | None, int | None]:
    """Pick the first processed film and scene 1 as the default anchor.

    "Processed" = ``is_processed`` flag from
    :func:`cinemateca.library.scan_library`, which derives from
    ``scene_count > 0``. Returns ``(None, None)`` when no film qualifies
    so the caller can render the empty-state branch.
    """
    slug = next((f.slug for f in films if getattr(f, "is_processed", False)), None)
    if slug is None:
        return None, None
    return slug, 1
