"""Cenas grid (``.c-cp``) context builder — Task 15.

Extracted verbatim from ``api/services/scenes_service.py`` (lines ~514–1051)
during the A1 decomposition (WS-2 Task 2). The private grouping/sorting
helpers live in ``_grouping.py``; the per-film traversal + card-conversion
helpers live in ``_film_grid.py``.
"""

from __future__ import annotations

import logging
from typing import Any

from api.contexts import CenasContext
from api.services.scenes._film_grid import (
    _build_groups_by_film,
    _format_runtime_hm,
)
from api.services.scenes._grouping import (
    _VALID_GROUPS,
    _VALID_SORTS,
)
from api.services.scenes._tipo import _VALID_BUCKETS

logger = logging.getLogger(__name__)


def build_cenas_context(
    cfg: Any,
    *,
    tags: list[str] | None = None,
    keyword: str = "",
    selected_scene_id: int | None = None,
    slug: str | None = None,
    group: str = "film",
    sort: str = "timecode",
    bucket: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> CenasContext:
    """Return the full Cenas-tab template context.

    Powers the ``/scenes`` full-page route and the ``/tab/scenes``
    fragment. The context shape matches what the new
    ``partials/scenes.html`` consumes:

      * ``groups_by_film`` — ordered list of per-film groups,
      * ``selected_scene_id`` — id of the card to mark ``.sel``,
      * ``total_scenes`` / ``film_count`` / ``total_runtime_str`` /
        ``total_keyframes_size`` — countrow summary,
      * ``visible_field_count`` / ``active_filter_count`` — toolrow pips,
      * ``no_data`` — true when no card was produced across all films,
        so the partial can render the empty-state hint instead of an
        empty grid (and the parity test against ``/tab/scenes`` keeps
        matching),
      * ``available_tags`` — union of normalized tag-index keys, kept
        for backwards-compat with legacy tag-filter callers,
      * ``cards`` — legacy flat-list shape retained so any code still
        reading ``ctx["cards"]`` (catalog tests, etc.) keeps working
        until Phase 3 cleanup removes the field.

    Aggregate-only (no per-film slug branch): the Cenas redesign always
    renders the library-wide grouped grid; the legacy per-film view
    falls out as a sidebar selection that filters the same grid in a
    later task. Until then, slug-aware callers can still apply
    keyword / tag filters via the existing ``build_scenes_grid`` path.
    """
    tags = list(tags or [])
    keyword = keyword or ""
    (
        groups,
        films,
        total_scenes,
        total_runtime_s,
        all_tags,
    ) = _build_groups_by_film(
        cfg,
        tags=tags,
        keyword=keyword,
        slug=slug,
        group=group,
        sort=sort,
        bucket=bucket,
    )

    # Flat cards list — preserves the legacy context key so older
    # template includes and tests that read ``cards`` directly do not
    # break during the transition. Pulls from each group's scenes so
    # the keyword/tag filter applied above is honoured. We iterate
    # under a different name so the outer ``group`` argument (used
    # again below for ``active_group``) isn't shadowed by the dict
    # element here.
    flat_cards: list[dict] = []
    for grp in groups:
        # Prefer each scene's own per-card film (group=tipo / group=none
        # can mix films within one group); fall back to the group's
        # film namespace for the group=film path where every scene
        # shares the heading's film.
        for s in grp["scenes"]:
            scene_film = s.get("film") or grp.get("film")
            film_slug = getattr(scene_film, "slug", None) or ""
            flat_cards.append({**s, "film_slug": film_slug})

    # ── A7: pagination ────────────────────────────────────────────────────────
    # ``total_scenes`` / ``film_count`` / ``total_runtime_s`` remain UNPAGED
    # so the countrow shows the honest library-wide totals ("showing N of M").
    # The grid is sliced by building a paged flat sequence, then filtering the
    # original groups down to only the scenes that fall in the current page.
    paged_cards = flat_cards[offset : offset + limit]
    paged_scene_ids: set[int] = {c["scene_id"] for c in paged_cards}

    # Rebuild groups_by_film to only contain paged scenes; drop empty groups.
    paged_groups: list[dict] = []
    for grp in groups:
        paged_scenes = [s for s in grp["scenes"] if s.get("scene_id") in paged_scene_ids]
        if paged_scenes:
            paged_groups.append({**grp, "scenes": paged_scenes})

    has_more = (offset + limit) < total_scenes
    has_prev = offset > 0

    return {
        "groups_by_film": paged_groups,
        "selected_scene_id": selected_scene_id,
        # UNPAGED totals — countrow shows library-wide summary.
        "total_scenes": total_scenes,
        "film_count": len(groups),
        "total_runtime_s": total_runtime_s,
        "total_runtime_str": _format_runtime_hm(total_runtime_s),
        # Pagination signals consumed by the grid partial's nav bar and sentinel.
        "has_more": has_more,
        "has_prev": has_prev,
        "current_offset": offset,
        "next_offset": offset + limit,
        "prev_offset": max(0, offset - limit),
        "current_limit": limit,
        # The keyframes-on-disk size is not summed today (would require
        # ``os.stat`` per keyframe — O(scenes) syscalls on every page
        # load). Em-dash placeholder until a cheap, cached source lands.
        "total_keyframes_size": "—",
        # First-paint toolrow counts. The fields control is client-side
        # localStorage, so the server emits the default visible count; Alpine
        # updates it after hydration if the user has hidden fields. The filter
        # count reflects active tag filters plus a left-pane bucket shortcut.
        "visible_field_count": 2,
        "active_filter_count": len(tags) + (1 if bucket in _VALID_BUCKETS else 0),
        "no_data": total_scenes == 0,
        "available_tags": sorted(all_tags),
        "cards": paged_cards,
        # Echo back the active query so a re-render preserves the input
        # value on full-page navigation (and the inspector route can
        # plumb it through later if needed).
        "query": keyword,
        # Active group/sort — surfaced so the toolrow's hidden inputs
        # and the radio popovers can seed their initial values on
        # full-page navigation. Normalised to a value in the
        # ``_VALID_*`` sets so the template never has to defend
        # against ``?group=foobar``.
        "active_group": group if group in _VALID_GROUPS else "film",
        "active_sort": sort if sort in _VALID_SORTS else "timecode",
        "active_bucket": bucket if bucket in _VALID_BUCKETS else "",
    }
