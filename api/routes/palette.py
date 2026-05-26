"""Command-palette search endpoint (Phase 7 / Task 27).

The palette is JS-driven (see ``web/static/js/palette.js``). The client
opens an overlay on ⌘K / Ctrl+K, debounces user input, and calls this
endpoint for grouped results. The endpoint returns a plain JSON payload
(no HTMX fragments) because the palette renders rows imperatively from
the array — neither htmx swap targets nor server-rendered partials are
useful here.

Shape of the response::

    {
      "navigate": [ {key, label, url, icon, kbd}, ... ],
      "actions":  [ {key, label, url, icon},      ... ],
      "films":    [ {key, label, sub, url, icon, slug}, ... ],
      "scenes_recent": [ {key, label, sub, url, icon, slug, scene_id}, ... ]
    }

All four groups are always present (even when empty) so the client's
group loop is deterministic.
"""

from __future__ import annotations

from fastapi import APIRouter

from api.deps import get_config
from api.services.palette_service import search_palette

router = APIRouter()


@router.get("/api/palette/search")
def palette_search(q: str = "") -> dict:
    """Return grouped palette results matching ``q``.

    Empty ``q`` returns the full static catalogues, every registered film,
    and a capped scene list (so the open-and-immediately-look pattern works
    without typing). A non-empty ``q`` filters each group by case-insensitive
    substring.

    Note: ``get_config`` is called directly (no FastAPI ``Depends``) so
    the conftest's dynamic per-module rebinding of ``get_config`` to the
    hermetic temp config takes effect under test. ``Depends(get_config)``
    would capture the original callable at import time and bypass the
    monkeypatch — every other route module in this codebase follows the
    same direct-call convention for the same reason.
    """
    cfg = get_config()
    return search_palette(cfg, q)
