"""Scene list and timeline context builders (multi-scene / Cenas tab)."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from api.services._scene_detail import build_inspector_context, tipo_of
from api.services.catalog import (
    build_cards,
    derive_fps,
    keyframe_url,
    load_json,
    load_metadata,
    to_smpte,
)
from cinemateca.library import FilmContext
from api.services.film_service import list_films

logger = logging.getLogger(__name__)


# ── Timeline helpers ──────────────────────────────────────────────────────────


def _hhmm_from_seconds(seconds: float) -> str:
    """Format seconds as HH:MM for timeline tick labels."""
    s = max(0, int(round(seconds)))
    hh = s // 3600
    mm = (s % 3600) // 60
    return f"{hh:02d}:{mm:02d}"


def _format_runtime_tc(runtime_s: float | None) -> str:
    """Return HH:MM:SS runtime tag; '--:--:--' when unknown."""
    if not runtime_s or runtime_s <= 0:
        return "--:--:--"
    s = int(round(runtime_s))
    hh = s // 3600
    mm = (s % 3600) // 60
    ss = s % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def _compute_timeline_ticks(runtime_s: float | None, count: int = 8) -> list[str]:
    """Return count evenly-spaced HH:MM tick labels across the runtime."""
    if not runtime_s or runtime_s <= 0 or count <= 0:
        return []
    step = runtime_s / count
    return [_hhmm_from_seconds(i * step) for i in range(count)]


def _build_scenes_for_timeline(
    kf_meta: list,
    data_dir: Path,
    fps: float,
    *,
    selected_scene_id: int | None,
    match_scene_ids: set[int],
) -> list[dict]:
    """Build the .scrub > .seg payload for every scene in the film."""
    scenes: list[dict] = []
    for entry in kf_meta:
        sid_raw = entry.get("scene_id")
        if sid_raw is None:
            continue
        try:
            sid = int(sid_raw)
        except (TypeError, ValueError):
            continue
        start_s = float(entry.get("start_time_s") or 0.0)
        timecode = to_smpte(start_s, fps) if start_s > 0 else ""
        scenes.append(
            {
                "id": sid,
                "scene_id": sid,
                "keyframe_url": keyframe_url(entry.get("filepath", ""), data_dir) or "",
                "timecode": timecode,
                "is_match": sid in match_scene_ids,
                "is_selected": selected_scene_id is not None and sid == selected_scene_id,
            }
        )
    return scenes


def build_timeline_context(
    cfg: Any,
    *,
    slug: str | None,
    scene_id: int | None,
    query: str = "",
) -> dict | None:
    """Build the bottom-timeline (.b-tl) context.

    Args:
        cfg: Loaded application config.
        slug: Film slug; None returns None.
        scene_id: Selected scene id; None returns None.
        query: Active search query string for state preservation.

    Returns:
        Context dict with selected_film, selected_scene, film_match_n, query;
        or None when the film/scene cannot be resolved.
    """
    if not slug or scene_id is None:
        return None

    inspector_ctx = build_inspector_context(cfg, scene_id=scene_id, slug=slug)
    if inspector_ctx is None:
        return None

    selected_scene = inspector_ctx["selected_scene"]
    film_obj = inspector_ctx["selected_film"]

    try:
        ctx = FilmContext.for_film(cfg, slug)
    except ValueError:
        return None

    kf_meta = load_json(ctx.metadata_dir / "keyframes_metadata.json") or []
    if not isinstance(kf_meta, list) or not kf_meta:
        return None

    fps = derive_fps(kf_meta)

    runtime_s: float | None = None
    for entry in reversed(kf_meta):
        end = float(entry.get("end_time_s") or 0.0)
        if end > 0:
            runtime_s = end
            break

    match_ids: set[int] = {scene_id}

    scenes_for_timeline = _build_scenes_for_timeline(
        kf_meta,
        ctx.data_dir,
        fps,
        selected_scene_id=scene_id,
        match_scene_ids=match_ids,
    )
    if not scenes_for_timeline:
        return None

    base_attrs = {
        "slug": slug,
        "title": getattr(film_obj, "title", None) or slug,
        "scene_count": getattr(film_obj, "scene_count", len(kf_meta)) or len(kf_meta),
        "year": getattr(film_obj, "year", None),
    }
    selected_film_ns = SimpleNamespace(
        **base_attrs,
        scenes_for_timeline=scenes_for_timeline,
        timeline_ticks=_compute_timeline_ticks(runtime_s),
        runtime_tc=_format_runtime_tc(runtime_s),
    )

    film_match_n = sum(1 for s in scenes_for_timeline if s["is_match"])

    return {
        "selected_film": selected_film_ns,
        "selected_scene": selected_scene,
        "film_match_n": film_match_n,
        "query": query,
    }


# ── Cenas grid helpers ────────────────────────────────────────────────────────


def _format_runtime_hm(seconds: float | None) -> str:
    """Return 'Xh Ym' for the countrow runtime summary; '—' when unknown."""
    if not seconds or seconds <= 0:
        return "—"
    s = int(round(seconds))
    hh = s // 3600
    mm = (s % 3600) // 60
    if hh == 0:
        return f"{mm}m"
    return f"{hh}h {mm:02d}m"


def _last_end_time_s(kf_meta: list) -> float:
    """Return the last positive end_time_s across a scene list, or 0.0."""
    for entry in reversed(kf_meta):
        try:
            end = float(entry.get("end_time_s") or 0.0)
        except (TypeError, ValueError):
            continue
        if end > 0:
            return end
    return 0.0


def _film_for_grid(film: Any, kf_meta: list) -> SimpleNamespace:
    """Wrap a Film in a namespace with grid-template attrs."""
    runtime_s = _last_end_time_s(kf_meta) if kf_meta else 0.0
    return SimpleNamespace(
        slug=film.slug,
        title=film.title,
        year=film.year,
        scene_count=film.scene_count if film.scene_count else len(kf_meta),
        director="",
        director_last="",
        runtime_tc=_format_runtime_hm(runtime_s),
        runtime_s=runtime_s,
    )


_VALID_GROUPS = frozenset({"film", "tipo", "none"})
_VALID_SORTS = frozenset({"timecode", "duration", "pins"})

# Stable display order for ``group=tipo`` headings — narrative arc that
# mirrors how a curator reads through a film. New ``tipo`` values added
# in ``tipo_of`` MUST also land here or they fall through alphabetically.
_TIPO_DISPLAY_ORDER = (
    "cartela",
    "interior",
    "exterior",
    "dialogo",
    "transicao",
)
_TIPO_LABEL = {
    "cartela": "Cartela",
    "interior": "Interior",
    "exterior": "Exterior",
    "dialogo": "Diálogo",
    "transicao": "Transição",
}


def _card_to_scene(card: dict, *, film_ns: SimpleNamespace | None = None) -> dict:
    """Convert a catalog build_cards dict to the grid template's scene shape.

    ``film_ns`` is the per-scene Film namespace needed for cross-film
    groupings (``group=tipo`` / ``group=none``). Legacy callers may omit it
    when every scene in a group shares the same film.
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
        "start_s": float(card.get("start_s") or 0.0),
        "duration_s": float(card.get("duration_s") or 0.0),
        "tipo": tipo_of(
            list(card.get("all_tags") or card.get("tags") or []),
            card.get("full_description") or card.get("description") or "",
        ),
        "pin_count": int(card.get("pin_count") or 0),
        "version": card.get("version") or None,
        "film": film_ns,
    }


def _sort_scenes(scenes: list[dict], sort: str) -> list[dict]:
    """Stable sort of scene dicts by the requested key.

    ``timecode`` → ``start_s`` ascending (default).
    ``duration`` → longest first.
    ``pins`` → ``pin_count`` descending, ties break by ``start_s``.
    """
    if sort == "duration":
        return sorted(scenes, key=lambda s: -float(s.get("duration_s") or 0.0))
    if sort == "pins":
        return sorted(
            scenes,
            key=lambda s: (
                -int(s.get("pin_count") or 0),
                float(s.get("start_s") or 0.0),
            ),
        )
    return sorted(scenes, key=lambda s: float(s.get("start_s") or 0.0))


def _regroup(
    per_film: list[tuple[SimpleNamespace, list[dict]]],
    *,
    group: str,
    sort: str,
) -> list[dict]:
    """Apply ``group`` + ``sort`` to the per-film scene lists.

    ``group=film`` — one group per film (default).
    ``group=tipo`` — one group per scene tipo across all films.
    ``group=none`` — one flat group, no visible heading.
    """
    if group == "none":
        flat_scenes: list[dict] = []
        for _f, scenes in per_film:
            flat_scenes.extend(scenes)
        flat_scenes = _sort_scenes(flat_scenes, sort)
        if not flat_scenes:
            return []
        first_film = flat_scenes[0].get("film") or (per_film[0][0] if per_film else None)
        return [
            {
                "film": first_film,
                "scenes": flat_scenes,
                "match_count": len(flat_scenes),
                "is_grouped": False,
                "heading_label": "",
                "heading_sub": "",
                "heading_total": 0,
                "heading_dot": "var(--c-accent)",
            }
        ]

    if group == "tipo":
        by_tipo: dict[str, list[dict]] = {}
        for _f, scenes in per_film:
            for s in scenes:
                tipo = str(s.get("tipo") or "transicao")
                by_tipo.setdefault(tipo, []).append(s)
        out: list[dict] = []
        seen_tipos: set[str] = set()
        ordered_tipos = list(_TIPO_DISPLAY_ORDER) + sorted(
            t for t in by_tipo if t not in _TIPO_DISPLAY_ORDER
        )
        for tipo in ordered_tipos:
            scenes = by_tipo.get(tipo) or []
            if not scenes or tipo in seen_tipos:
                continue
            seen_tipos.add(tipo)
            sorted_scenes = _sort_scenes(scenes, sort)
            label = _TIPO_LABEL.get(tipo, tipo.capitalize())
            tipo_ns = SimpleNamespace(
                slug=tipo,
                title=label,
                year=None,
                director=None,
                director_last=None,
                scene_count=len(scenes),
                runtime_tc="",
                runtime_s=0.0,
            )
            out.append(
                {
                    "film": tipo_ns,
                    "scenes": sorted_scenes,
                    "match_count": len(sorted_scenes),
                    "is_grouped": True,
                    "heading_label": label,
                    "heading_sub": "",
                    "heading_total": len(sorted_scenes),
                    "heading_dot": f"var(--c-cat-{tipo})",
                }
            )
        return out

    # Default: group=film
    out = []
    for film_ns, scenes in per_film:
        sorted_scenes = _sort_scenes(scenes, sort)
        sub_parts: list[str] = []
        if film_ns.year:
            sub_parts.append(str(film_ns.year))
        if film_ns.director:
            sub_parts.append(film_ns.director)
        out.append(
            {
                "film": film_ns,
                "scenes": sorted_scenes,
                "match_count": len(sorted_scenes),
                "is_grouped": True,
                "heading_label": film_ns.title,
                "heading_sub": " · ".join(sub_parts),
                "heading_total": film_ns.scene_count,
                "heading_dot": "var(--c-accent)",
            }
        )
    return out


def _build_groups_by_film(
    cfg: Any,
    *,
    tags: list[str],
    keyword: str,
    slug: str | None = None,
    group: str = "film",
    sort: str = "timecode",
) -> tuple[list[dict], list[Any], int, float, set[str]]:
    """Walk the library and produce the groups_by_film template payload.

    Args:
        cfg: Loaded application config.
        tags: Tag filters to apply.
        keyword: Text keyword filter.
        slug: When set, restrict to one film.
        group: Grouping mode — "film" (default), "tipo", or "none".
        sort: Sort key — "timecode" (default), "duration", or "pins".

    Returns:
        Tuple of (groups, films, total_scenes, total_runtime_s, all_tags).
    """
    library_dir = Path(cfg.paths.library_dir)
    films: list[Any] = []
    total_scenes = 0
    total_runtime_s = 0.0
    all_tags: set[str] = set()

    if group not in _VALID_GROUPS:
        group = "film"
    if sort not in _VALID_SORTS:
        sort = "timecode"

    all_films = list(list_films(library_dir))
    if slug is not None:
        if not any(f.slug == slug for f in all_films):
            FilmContext.for_film(cfg, slug)
        all_films = [f for f in all_films if f.slug == slug]

    per_film: list[tuple[SimpleNamespace, list[dict]]] = []
    for film in all_films:
        films.append(film)
        try:
            ctx = FilmContext.for_film(cfg, film.slug)
        except ValueError:
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
            continue
        film_ns = _film_for_grid(film, kf_meta)
        scenes = [_card_to_scene(c, film_ns=film_ns) for c in cards]
        total_scenes += len(scenes)
        total_runtime_s += film_ns.runtime_s
        per_film.append((film_ns, scenes))

    groups = _regroup(per_film, group=group, sort=sort)
    return groups, films, total_scenes, total_runtime_s, all_tags


def build_cenas_context(
    cfg: Any,
    *,
    tags: list[str] | None = None,
    keyword: str = "",
    selected_scene_id: int | None = None,
    slug: str | None = None,
    group: str = "film",
    sort: str = "timecode",
) -> dict:
    """Return the full Cenas-tab template context.

    Args:
        cfg: Loaded application config.
        tags: Tag filters.
        keyword: Text keyword filter.
        selected_scene_id: Scene id to mark as selected in the grid.
        slug: When set, restrict the grid to one film.
        group: Grouping mode — "film" (default), "tipo", or "none".
        sort: Sort key — "timecode" (default), "duration", or "pins".

    Returns:
        Context dict for the scenes.html / tab/scenes partial.
    """
    tags = list(tags or [])
    keyword = keyword or ""
    (
        groups,
        films,
        total_scenes,
        total_runtime_s,
        all_tags,
    ) = _build_groups_by_film(cfg, tags=tags, keyword=keyword, slug=slug, group=group, sort=sort)

    flat_cards: list[dict] = []
    for grp in groups:
        for s in grp["scenes"]:
            scene_film = s.get("film") or grp.get("film")
            film_slug = getattr(scene_film, "slug", None) or ""
            flat_cards.append({**s, "film_slug": film_slug})

    return {
        "groups_by_film": groups,
        "selected_scene_id": selected_scene_id,
        "total_scenes": total_scenes,
        "film_count": len(groups),
        "total_runtime_s": total_runtime_s,
        "total_runtime_str": _format_runtime_hm(total_runtime_s),
        "total_keyframes_size": "—",
        "visible_field_count": 2,
        "active_filter_count": len(tags),
        "no_data": total_scenes == 0,
        "available_tags": sorted(all_tags),
        "cards": flat_cards,
        "query": keyword,
        "active_group": group if group in _VALID_GROUPS else "film",
        "active_sort": sort if sort in _VALID_SORTS else "timecode",
    }
