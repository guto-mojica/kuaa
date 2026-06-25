"""Per-film metadata loading and card-conversion helpers for the Cenas grid.

Peeled from ``api/services/scenes/_cards.py`` (which was itself extracted
from ``api/services/scenes_service.py``) during the A1 decomposition (WS-2
Task 2). Contains the library-traversal loop, runtime formatting, and the
``_card_to_scene`` shape conversion — all the I/O-adjacent helpers that
``build_cenas_context`` delegates to.

Public names: ``_format_runtime_hm``, ``_build_groups_by_film``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from api.services.catalog import (
    build_cards,
    load_metadata,
)
from api.services.scenes._grouping import (
    _VALID_GROUPS,
    _VALID_SORTS,
    _regroup,
)
from api.services.scenes._tipo import (
    _VALID_BUCKETS,
    tipo_of,
)
from kuaa.library import FilmContext, scan_library

logger = logging.getLogger(__name__)


def _format_runtime_hm(seconds: float | None) -> str:
    """Return ``"Xh Ym"`` for the countrow runtime summary.

    Empty / non-positive durations collapse to ``"—"`` (em-dash) so the
    countrow's ``<span class="v">`` renders a typographic placeholder
    instead of ``0h 0m``. Used by both the per-film ``runtime_tc`` and
    the library-wide ``total_runtime_str``.
    """
    if not seconds or seconds <= 0:
        return "—"  # em-dash
    s = int(round(seconds))
    hh = s // 3600
    mm = (s % 3600) // 60
    if hh == 0:
        return f"{mm}m"
    return f"{hh}h {mm:02d}m"


def _last_end_time_s(kf_meta: list) -> float:
    """Return the last positive ``end_time_s`` across a scene list.

    Used to derive a per-film runtime estimate (the actual video may
    extend a few seconds past the last detected scene, but for the
    countrow's coarse summary this is indistinguishable from the truth).
    Returns ``0.0`` when no entry exposes a positive end time.
    """
    for entry in reversed(kf_meta):
        try:
            end = float(entry.get("end_time_s") or 0.0)
        except (TypeError, ValueError):
            continue
        if end > 0:
            return end
    return 0.0


def _film_for_grid(film: Any, kf_meta: list) -> SimpleNamespace:
    """Wrap a ``Film`` in a namespace exposing the grid template's attrs.

    The ``Film`` dataclass has no ``director`` / ``runtime_tc`` /
    ``director_last`` fields (see ``kuaa.library.Film``), but the
    grid template reads all three. Rather than widen the dataclass for
    presentational concerns, this returns a ``SimpleNamespace`` that:

      * mirrors the registered slug / title / year / scene_count;
      * derives ``runtime_tc`` from the film's keyframe metadata;
      * exposes ``director`` / ``director_last`` (both empty today —
        the registry does not store a director field).

    If/when the registry gains a director column, populate it here and
    the template picks it up without further changes.
    """
    runtime_s = _last_end_time_s(kf_meta) if kf_meta else 0.0
    runtime_tc = _format_runtime_hm(runtime_s)
    # Director is not in films.json today; reserved for a future
    # metadata extension. Empty strings collapse the ``· {director}``
    # span in the template via its ``{% if group.film.director %}`` guard.
    director = ""
    director_last = ""
    return SimpleNamespace(
        slug=film.slug,
        title=film.title,
        year=film.year,
        scene_count=film.scene_count if film.scene_count else len(kf_meta),
        director=director,
        director_last=director_last,
        runtime_tc=runtime_tc,
        runtime_s=runtime_s,
    )


def _card_to_scene(card: dict, *, film_ns: SimpleNamespace | None = None) -> dict:
    """Convert a catalog ``build_cards`` dict to the grid template's scene shape.

    Adds the keys the new template reads (``id``, ``slug``, ``tipo``,
    ``pin_count``, ``version``, ``keyframe_url``, ``start_s``,
    ``duration_s``, ``film``) while preserving the original
    ``scene_id`` / ``timecode`` keys so any downstream code that still
    consumes the catalog shape keeps working.

    ``slug`` here is the *scene* slug shown on the card body (e.g.
    ``"scene 351"``) — distinct from the *film* slug carried by the
    enclosing group. Without a stable scene-slug source in the metadata
    today it falls back to ``f"scene {scene_id}"``; later phases can
    swap in a curated scene title (e.g. from a description's first
    clause) without touching the template.

    ``film_ns`` is the per-scene Film namespace (slug, title, year,
    director_last) used by the scenes_grid template's per-card sub
    line + hx-get URL. When omitted (legacy callers) the caller is
    responsible for inferring the film from the enclosing group —
    which works for ``group=film`` only. New callers MUST pass it so
    ``group=tipo`` / ``group=none`` can mix scenes across films and
    each card still resolves its own inspector URL.
    """
    sid = card.get("scene_id")
    try:
        sid_int = int(sid) if sid is not None else 0
    except (TypeError, ValueError):
        sid_int = 0
    return {
        "id": sid_int,
        "scene_id": sid_int,
        "slug": f"scene {sid_int}",
        "keyframe_url": card.get("img_url") or "",
        "timecode": card.get("timecode") or "",
        # Raw seconds — surfaced for the Cenas grid's Sort-by-Duration
        # path; the SMPTE ``timecode`` field stays the human-readable
        # render. ``start_s`` is also the canonical key for
        # Sort-by-Timecode (more precise than the truncated string).
        "start_s": float(card.get("start_s") or 0.0),
        "duration_s": float(card.get("duration_s") or 0.0),
        "tipo": tipo_of(
            list(card.get("all_tags") or card.get("tags") or []),
            card.get("full_description") or card.get("description") or "",
        ),
        "pin_count": int(card.get("pin_count") or 0),
        # ``version`` (V1/V2) is reserved for the multi-cut workflow
        # that lands in a later plan; the template hides the ``.ver``
        # pill when this is falsy.
        "version": card.get("version") or None,
        # Per-scene film namespace — needed for cross-film groupings
        # (``group=tipo`` / ``group=none``) where the enclosing group
        # heading no longer carries a single film identity.
        "film": film_ns,
    }


def _build_groups_by_film(
    cfg: Any,
    *,
    tags: list[str],
    keyword: str,
    slug: str | None = None,
    group: str = "film",
    sort: str = "timecode",
    bucket: str | None = None,
) -> tuple[list[dict], list[Any], int, float, set[str]]:
    """Walk the library and produce the ``groups_by_film`` template payload.

    Returns ``(groups, films, total_scenes, total_runtime_s, all_tags)``.
    ``group`` ∈ {"film", "tipo", "none"}, ``sort`` ∈ {"timecode",
    "duration", "pins"}, ``bucket`` narrows to one tipo. See
    ``build_cenas_context`` for the full contract.
    """
    library_dir = Path(cfg.paths.library_dir)
    films: list[Any] = []
    total_scenes = 0
    total_runtime_s = 0.0
    all_tags: set[str] = set()

    # Normalise unknown values silently — a stray ?group=foobar query
    # shouldn't 5xx the page. Default to the safest option in each case.
    if group not in _VALID_GROUPS:
        group = "film"
    if sort not in _VALID_SORTS:
        sort = "timecode"
    if bucket not in _VALID_BUCKETS:
        bucket = None

    # Per-film slug filter (sidebar-driven): only render the matching
    # film's group. The aggregate path still walks the whole library so
    # the empty-state hint stays accurate when the registered slug has
    # no on-disk metadata. ``ValueError`` from ``FilmContext.for_film``
    # surfaces to the caller (matches the legacy contract — the routes
    # use it to 4xx unknown slugs in HTMX-fetch paths).
    all_films = list(scan_library(library_dir))
    if slug is not None:
        if not any(f.slug == slug for f in all_films):
            # Trigger the same ValueError the legacy single-film path
            # produced via ``FilmContext.for_film(cfg, slug)``. Tests
            # pin this contract (``test_tab_scenes_unknown_slug_raises``).
            FilmContext.for_film(cfg, slug)
        all_films = [f for f in all_films if f.slug == slug]

    # Phase 1: build a flat ``(film_ns, [scene_dict])`` list per film.
    # Phase 2 below regroups + sorts based on ``group`` / ``sort``.
    per_film: list[tuple[SimpleNamespace, list[dict]]] = []
    for film in all_films:
        films.append(film)
        try:
            ctx = FilmContext.for_film(cfg, film.slug)
        except ValueError:
            # Slug registered but disk layout missing — skip cleanly.
            continue
        kf_meta, desc_by_scene, vis_by_scene, tag_index = load_metadata(ctx.metadata_dir)
        all_tags.update(tag_index.keys())
        cards = build_cards(
            kf_meta,
            desc_by_scene,
            vis_by_scene,
            tag_index,
            ctx.data_dir,
            tags,
            keyword,
        )
        if not cards:
            # Per-film empty result after filtering — don't emit a
            # heading the user can't drill into.
            continue
        film_ns = _film_for_grid(film, kf_meta)
        scenes = [_card_to_scene(c, film_ns=film_ns) for c in cards]
        if bucket:
            scenes = [s for s in scenes if s.get("tipo") == bucket]
        if not scenes:
            continue
        total_scenes += len(scenes)
        total_runtime_s += film_ns.runtime_s
        per_film.append((film_ns, scenes))

    groups = _regroup(per_film, group=group, sort=sort)
    return groups, films, total_scenes, total_runtime_s, all_tags
