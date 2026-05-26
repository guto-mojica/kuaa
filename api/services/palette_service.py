"""Command-palette result aggregation (Phase 7 / Task 27).

The palette is a single global keyboard-driven entry point (⌘K / Ctrl+K) for
navigation + cross-tab actions + library search. The service groups results
into stable categories the client renders in fixed order:

  * ``navigate`` — main tab destinations (Home, Search, Scenes, Annotate,
    Rhymes, Processing). The hotkeys ("1".."5") match the keyboard help
    overlay (Task 28).
  * ``actions`` — backed global commands (locale switch, about). Distinct
    from ``navigate`` because they may trigger server-side state changes.
  * ``films`` — registered library films, filtered by label.
  * ``scenes_recent`` — recent/first processed scenes from registered films,
    filtered by scene label or description.

The filter is intentionally simple (case-insensitive substring match
against ``label``). Fuzzy matching + weighting land in the same later
phase that wires recent scenes.
"""

from __future__ import annotations

from typing import Any

from cinemateca import library
from cinemateca.scene_ids import scene_id_key

# Static catalogues. Defined at module scope so they are NOT rebuilt per
# request; the client filters them locally after one fetch, so payload size
# is the cost we trade off, not allocation.
NAVIGATE: list[dict[str, Any]] = [
    {"key": "go-home", "label": "Home", "url": "/", "icon": "home", "kbd": ""},
    {"key": "go-buscar", "label": "Search", "url": "/search", "icon": "search", "kbd": "1"},
    {"key": "go-cenas", "label": "Scenes", "url": "/scenes", "icon": "grid", "kbd": "2"},
    {"key": "go-anotar", "label": "Annotate", "url": "/annotate", "icon": "tag", "kbd": "3"},
    {"key": "go-rimas", "label": "Rhymes", "url": "/rimas", "icon": "rhymes", "kbd": "4"},
    {"key": "go-proc", "label": "Processing", "url": "/processing", "icon": "proc", "kbd": "5"},
]

ACTIONS: list[dict[str, Any]] = [
    {"key": "switch-pt", "label": "Switch to PT-BR", "icon": "globe", "url": "/api/locale/pt_BR"},
    {"key": "switch-en", "label": "Switch to English", "icon": "globe", "url": "/api/locale/en"},
    {"key": "about", "label": "About Cinemateca", "icon": "doc", "url": "/about"},
]


def _matches(item: dict[str, Any], qn: str) -> bool:
    """Case-insensitive substring filter against the item label.

    Empty/whitespace queries match everything (used by the initial open
    render before the user has typed). The match is keyed off ``label``
    only — ``sub`` and ``key`` are deliberately excluded so noise like a
    scene-count digit can't pull an unrelated film into the results.
    """
    if not qn:
        return True
    return qn in item["label"].lower()


def _description_map(metadata_dir) -> dict[str, str]:
    raw = library.load_json(metadata_dir / "scene_descriptions.json") or []
    if not isinstance(raw, list):
        return {}
    out: dict[str, str] = {}
    for row in raw:
        if not isinstance(row, dict) or "scene_id" not in row:
            continue
        out[scene_id_key(row["scene_id"])] = str(row.get("description") or "")
    return out


def _scene_matches(item: dict[str, Any], qn: str) -> bool:
    if not qn:
        return True
    return qn in f"{item.get('label', '')} {item.get('sub', '')}".lower()


def _scene_results(cfg: Any, films_data: list[Any], qn: str, limit: int = 8) -> list[dict[str, Any]]:
    """Return cheap palette scene rows from existing per-film metadata."""
    rows: list[dict[str, Any]] = []
    for film in films_data:
        try:
            ctx = library.FilmContext.for_film(cfg, film.slug)
        except ValueError:
            continue
        kf_meta = library.load_json(ctx.metadata_dir / "keyframes_metadata.json") or []
        if not isinstance(kf_meta, list):
            continue
        desc_by_scene = _description_map(ctx.metadata_dir)
        for entry in kf_meta:
            if not isinstance(entry, dict) or "scene_id" not in entry:
                continue
            try:
                sid = int(entry["scene_id"])
            except (TypeError, ValueError):
                continue
            sid_key = scene_id_key(sid)
            timecode = entry.get("timecode_start") or entry.get("start_timecode") or ""
            desc = desc_by_scene.get(sid_key, "")
            sub_bits = [film.title]
            if timecode:
                sub_bits.append(str(timecode))
            if desc:
                sub_bits.append(desc[:96])
            item: dict[str, Any] = {
                "key": f"scene-{film.slug}-{sid}",
                "label": f"Scene {sid:03d}",
                "sub": " · ".join(sub_bits),
                "url": f"/scenes?film={film.slug}&scene={sid}",
                "icon": "grid",
                "badge": "scene",
                "slug": film.slug,
                "scene_id": sid,
            }
            if _scene_matches(item, qn):
                rows.append(item)
                if len(rows) >= limit:
                    return rows
    return rows


def search_palette(cfg, q: str) -> dict[str, list[dict[str, Any]]]:
    """Return grouped palette results for query string ``q``.

    The empty-q response returns every static row plus all registered
    films — this is the panel the user sees on first open. Non-empty
    queries filter each group independently; an empty group is still
    present in the response so the client renders a stable JSON shape.

    ``scenes_recent`` is capped so the palette stays light while still
    making the "Search films, scenes, actions" promise true.
    """
    qn = (q or "").strip().lower()

    navigate = [x for x in NAVIGATE if _matches(x, qn)]
    actions = [x for x in ACTIONS if _matches(x, qn)]

    films_data = library.scan_library(cfg.paths.library_dir)
    films: list[dict[str, Any]] = []
    for f in films_data:
        year_str = str(f.year) if f.year else ""
        sub_bits = [b for b in (year_str, f"{f.scene_count} scenes") if b]
        item: dict[str, Any] = {
            "key": f"film-{f.slug}",
            "label": f.title,
            "sub": " · ".join(sub_bits),
            "url": f"/scenes?film={f.slug}",
            "icon": "film",
            "slug": f.slug,
        }
        if _matches(item, qn):
            films.append(item)

    scenes_recent = _scene_results(cfg, films_data, qn)

    return {
        "navigate": navigate,
        "actions": actions,
        "films": films,
        "scenes_recent": scenes_recent,
    }
