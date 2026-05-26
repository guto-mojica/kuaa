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
  * ``scenes_recent`` — placeholder; populated in a later phase once a
    cheap per-film scene index exists. Empty for now so the empty-q
    response stays O(films).

The filter is intentionally simple (case-insensitive substring match
against ``label``). Fuzzy matching + weighting land in the same later
phase that wires recent scenes.
"""

from __future__ import annotations

from typing import Any

from cinemateca import library

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


def search_palette(cfg, q: str) -> dict[str, list[dict[str, Any]]]:
    """Return grouped palette results for query string ``q``.

    The empty-q response returns every static row plus all registered
    films — this is the panel the user sees on first open. Non-empty
    queries filter each group independently; an empty group is still
    present in the response so the client renders a stable JSON shape.

    ``scenes_recent`` is always returned for shape stability but stays
    empty until the per-film scene fuzzy index lands.
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

    # Scenes_recent: placeholder for a later phase. Skipping the per-film
    # description scan on every keystroke is deliberate — the palette is
    # supposed to feel instant.
    scenes_recent: list[dict[str, Any]] = []

    return {
        "navigate": navigate,
        "actions": actions,
        "films": films,
        "scenes_recent": scenes_recent,
    }
