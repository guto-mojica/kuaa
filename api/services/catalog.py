"""Catalog service — scene-card and scenes-tab template builders.

Path/URL/timecode utilities and per-film metadata loaders live in
``cinemateca.library.{paths,metadata}`` and are re-exported below so
existing call sites (``scenes_service``, ``annotations``, ``rhymes_service``,
etc.) keep importing them from ``api.services.catalog``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from cinemateca.library import FilmContext, scan_library
from cinemateca.library.metadata import (  # noqa: F401
    load_metadata,
    load_tag_index,
)
from cinemateca.library.paths import (  # noqa: F401
    derive_fps,
    keyframe_url,
    load_json,
    to_smpte,
)
from cinemateca.scene_ids import scene_id_key

logger = logging.getLogger(__name__)


# ── Scene-card construction ───────────────────────────────────────────────────


def _select_tags_by_frequency(scene_tags: list[str], tag_index: dict, n: int = 16) -> list[str]:
    """Return up to n tags sampled across the global-frequency spectrum.

    Sorts ascending by corpus frequency (rare → common). If the scene has
    > n tags, picks n at evenly-spaced positions so the selection spans
    rare scene-specific labels to common generic ones, rather than taking
    the first n alphabetically.
    """
    by_freq = sorted(scene_tags, key=lambda t: len(tag_index.get(t, [])))
    if len(by_freq) <= n:
        return by_freq
    total = len(by_freq)
    return [by_freq[i * total // n] for i in range(n)]


def build_cards(
    kf_meta: list,
    desc_by_scene: dict,
    vis_by_scene: dict,
    tag_index: dict,
    data_dir: Path,
    selected_tags: list[str],
    keyword: str,
) -> list[dict]:
    """Filter ``kf_meta`` and build scene-card dicts for the template.

    Tag filter: intersect scene_ids across all selected tags. Keyword
    filter: search the description text blob. Tags on each card are
    selected by :func:`_select_tags_by_frequency` (up to 16, sampled
    across the frequency spectrum). ``tag_index`` is expected normalized
    (as :func:`load_metadata` returns it).
    """
    scenes = list(kf_meta)
    fps = derive_fps(kf_meta)

    # Tag filter — intersect scene_ids across all selected tags.
    # tag_index is already normalized to {tag: {canonical str id}}, so
    # the membership test is str-vs-str.
    if selected_tags and tag_index:
        valid_ids = set(tag_index.get(selected_tags[0], set()))
        for tag in selected_tags[1:]:
            valid_ids &= set(tag_index.get(tag, set()))
        scenes = [s for s in scenes if scene_id_key(s.get("scene_id", "")) in valid_ids]

    # Keyword filter — search description text blob
    if keyword:
        kw = keyword.lower()
        filtered = []
        for s in scenes:
            sid = scene_id_key(s.get("scene_id", ""))
            desc = desc_by_scene.get(sid, {})
            blob = " ".join(str(v) for v in desc.values()).lower()
            if kw in blob:
                filtered.append(s)
        scenes = filtered

    # Deduplicate by scene_id — the detector writes N keyframes per scene
    # (N=3 by default) for embedding density. The scene browser shows one
    # card per scene; search deduplicates at query time via max(similarity).
    seen_scene_ids: set = set()
    unique_scenes: list = []
    for s in scenes:
        sid = scene_id_key(s.get("scene_id", ""))
        if sid not in seen_scene_ids:
            seen_scene_ids.add(sid)
            unique_scenes.append(s)
    scenes = unique_scenes

    cards = []
    for s in scenes:
        sid = scene_id_key(s.get("scene_id", ""))
        fp = Path(s.get("filepath", ""))
        img_url = keyframe_url(fp, data_dir)
        start_s = float(s.get("start_time_s") or 0.0)
        if start_s > 0:
            tc = to_smpte(start_s, fps)
        else:
            tc = s.get("timecode_start") or s.get("start_timecode", "")

        # Tags from tag_index (inverted lookup). tag_index ids are
        # already canonical str keys, so this is direct str-vs-str.
        all_scene_tags = list({tag for tag, ids in tag_index.items() if sid in ids})
        all_tags_sorted = sorted(all_scene_tags, key=lambda t: len(tag_index.get(t, [])))
        scene_tags = _select_tags_by_frequency(all_scene_tags, tag_index)

        # Visual analysis summary
        vis = vis_by_scene.get(sid, {})
        env = vis.get("environment", {})
        env_parts = [p for p in [env.get("location", ""), env.get("time_of_day", "")] if p]
        num_people = vis.get("num_faces")

        # Description one-liner
        desc = desc_by_scene.get(sid, {})
        description = desc.get("description") or ""

        # Duration is end_time_s − start_time_s, both already in
        # ``s`` (keyframe metadata). We carry the raw seconds so the
        # Cenas grid's "Sort by Duration" can compare without parsing
        # the SMPTE string back into seconds.
        end_s = float(s.get("end_time_s") or 0.0)
        duration_s = max(0.0, end_s - start_s)
        cards.append(
            {
                "scene_id": s.get("scene_id"),
                "img_url": img_url,
                "timecode": tc,
                "start_s": start_s,
                "duration_s": duration_s,
                "tags": scene_tags,
                "all_tags": all_tags_sorted,
                "environment": " · ".join(env_parts),
                "num_people": num_people,
                "description": description[:120] if description else "",
                "full_description": description,
            }
        )

    return cards


# ── Tab context builders ──────────────────────────────────────────────────────


def build_scenes_context(ctx: FilmContext) -> dict:
    """Build the scenes-tab template context (no tag/keyword filter).

    Shared by the ``/tab/scenes`` HTMX fragment and the ``/scenes``
    full-page route so both render identical markup, including the
    empty-state hint when no keyframes exist. Keys: ``cards``,
    ``available_tags``, ``no_data``.
    """
    kf_meta, desc_by_scene, vis_by_scene, tag_index = load_metadata(ctx.metadata_dir)
    available_tags = sorted(tag_index.keys())
    cards = build_cards(kf_meta, desc_by_scene, vis_by_scene, tag_index, ctx.data_dir, [], "")
    return {
        "cards": cards,
        "available_tags": available_tags,
        "no_data": not kf_meta,
    }


def build_scenes_grid(ctx: FilmContext, tags: list[str], keyword: str) -> dict:
    """Build the filtered scenes-grid context for ``/api/scenes``.

    Same single key (``cards``) the ``scenes_grid.html`` partial
    consumes; behaviour identical to the prior inline route body.
    """
    kf_meta, desc_by_scene, vis_by_scene, tag_index = load_metadata(ctx.metadata_dir)
    cards = build_cards(
        kf_meta, desc_by_scene, vis_by_scene, tag_index, ctx.data_dir, tags, keyword
    )
    return {"cards": cards}


def build_scenes_grid_aggregate(cfg: Any, tags: list[str], keyword: str) -> dict:
    """Build the filtered scenes-grid context across ALL films.

    Filter-aware sibling of :func:`build_scenes_context_aggregate` for
    the ``/api/scenes`` grid-refresh endpoint. Walks the library and
    applies *tags* and *keyword* per film via :func:`build_cards`. Each
    card is annotated with ``film_slug`` / ``film_title`` for template
    grouping. Returns ``{"cards": all_cards}`` — matching the per-film
    return shape so the same partial works in both modes.
    """
    library_dir = Path(cfg.paths.library_dir)
    all_cards: list[dict] = []
    for film in scan_library(library_dir):
        ctx = FilmContext.for_film(cfg, film.slug)
        kf_meta, desc_by_scene, vis_by_scene, tag_index = load_metadata(ctx.metadata_dir)
        cards = build_cards(
            kf_meta,
            desc_by_scene,
            vis_by_scene,
            tag_index,
            ctx.data_dir,
            tags,
            keyword,
        )
        for c in cards:
            c["film_slug"] = film.slug
            c["film_title"] = film.title
        all_cards.extend(cards)
    return {"cards": all_cards}


def build_scenes_context_aggregate(cfg: Any) -> dict:
    """Build the scenes context across ALL films in the library.

    For each film: load per-film metadata, build cards (path math via
    ``FilmContext.for_film``), annotate each card with ``film_slug`` +
    ``film_title``, and concatenate. Tolerates registered-but-unprocessed
    films (no ``metadata/`` dir): ``load_metadata`` returns empty
    containers and the film contributes zero cards.

    ``available_tags`` is the union of per-film tag-index keys.
    ``no_data`` is True iff no card was produced across all films —
    diverges intentionally from ``build_scenes_context``'s ``not kf_meta``
    test (aggregate "no data" means the whole library is empty, not one
    film). Loads all per-film metadata from disk on every call; large
    libraries (~100+ films) may want a request-scoped cache.
    """
    library_dir = Path(cfg.paths.library_dir)
    all_cards: list[dict] = []
    all_tags: set[str] = set()
    for film in scan_library(library_dir):
        ctx = FilmContext.for_film(cfg, film.slug)
        kf_meta, desc_by_scene, vis_by_scene, tag_index = load_metadata(ctx.metadata_dir)
        cards = build_cards(kf_meta, desc_by_scene, vis_by_scene, tag_index, ctx.data_dir, [], "")
        for c in cards:
            c["film_slug"] = film.slug
            c["film_title"] = film.title
        all_cards.extend(cards)
        all_tags.update(tag_index.keys())
    return {
        "cards": all_cards,
        "available_tags": sorted(all_tags),
        "no_data": not all_cards,
    }
