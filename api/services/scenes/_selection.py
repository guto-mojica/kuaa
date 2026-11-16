"""``?sel=`` parsing and filtering for the Scenes page.

A selection made in Search or Scenes travels as ``<film_slug>/<scene_id>``
pairs. ``scene_id`` is only unique within a film, so the slug is part of the
key.
"""

from __future__ import annotations

import re
from typing import Any

# Matches the cap the browser store enforces; it bounds the one-page render.
MAX_SELECTION = 200
_SLUG = re.compile(r"^[a-z0-9_]+$")


def parse_selection(raw: str | None) -> list[tuple[str, int]]:
    """Parse ``"a/1,b/2"`` into ``[("a", 1), ("b", 2)]``.

    Malformed parts and repeats are dropped (first occurrence wins) and at
    most ``MAX_SELECTION`` pairs are kept.
    """
    out: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for part in (raw or "").split(","):
        slug, sep, sid = part.partition("/")
        slug = slug.strip().lower()
        sid = sid.strip()
        if not sep or not _SLUG.match(slug) or not sid.isascii() or not sid.isdigit():
            continue
        key = (slug, int(sid))
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
        if len(out) == MAX_SELECTION:
            break
    return out


def format_selection(selection: list[tuple[str, int]]) -> str:
    """Inverse of :func:`parse_selection`, for carrying ``sel`` in form fields."""
    return ",".join(f"{slug}/{sid}" for slug, sid in selection)


def filter_groups_to_selection(
    groups: list[dict[str, Any]], selection: list[tuple[str, int]]
) -> list[dict[str, Any]]:
    """Keep only the selected scenes of each group; drop groups left empty.

    A scene's film is its own ``film`` namespace when present (groups by type
    mix films), else the group's.
    """
    wanted = set(selection)
    kept: list[dict[str, Any]] = []
    for grp in groups:
        scenes = []
        for s in grp["scenes"]:
            film = s.get("film") or grp.get("film")
            if (getattr(film, "slug", ""), s.get("scene_id")) in wanted:
                scenes.append(s)
        if scenes:
            kept.append({**grp, "scenes": scenes, "match_count": len(scenes)})
    return kept
