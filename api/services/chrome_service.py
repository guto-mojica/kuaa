"""Chrome context builder — Mojica TopBar + IconRail + LeftPane.

Phase-1 / Task 8 introduces a single source of truth for the Mojica
chrome context: the variables consumed by ``_topbar.html``,
``_icon_rail.html``, ``_left_pane.html`` and ``_left_pane_body.html``.

Before Task 8 every full-page route built its own (sparse) chrome bag
in ``api/server.py::render_page`` and ``api/deps.make_ctx`` defaulted
the topbar keys (`active_job_count`, `viewers`, `notification_count`,
`current_user`). Task 8 keeps those defaults — for routes that bypass
``render_page`` (HTMX tab fragments, partial endpoints) — but lifts the
real values into one builder so the new shell stays consistent across
routes and the values can grow (real viewers, real match percentages)
without touching every caller.

What this module does NOT do:

  * It does not load real per-film match percentages — that data lives
    in the (Month-2) search/Rimas pipeline and depends on a query, so
    ``film_match_pct`` ships as an empty dict for Phase 1. Search will
    overlay its own values when it lands.
  * It does not source real viewers / a real notification feed — those
    belong to the (later) collaboration epic; ``viewers``/
    ``notification_count``/``current_user`` ship empty.
  * It does not persist collections — the five "default collections"
    are a curated static list (the Mojica prototype's
    Coleções/Compartilhados section) so the LeftPane renders the design
    surface even before the persistence layer exists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from api.jobs import active_jobs
from api.services.film_service import list_films
from cinemateca.library import Film, LibraryState


class ChromeContext(TypedDict):
    """Typed shape of :func:`build_chrome_context`'s return value.

    Each key mirrors a Mojica chrome partial's expected variable; see the
    function docstring for per-key semantics.
    """

    films: list[Film]
    library_state: LibraryState
    total_runtime_minutes: int
    current_slug: str | None
    active_job_slugs: set[str]
    active_job_count: int
    film_match_pct: dict[str, int]
    film_match_counts: dict[str, int]
    collections: list[dict[str, Any]]
    viewers: list[dict[str, Any]]
    current_user: dict[str, Any] | None
    notification_count: int


def _default_collections(films: list[Film], total_scenes: int) -> list[dict[str, Any]]:
    """Return the curated default-collection list rendered in the LeftPane.

    Five entries match the Mojica prototype's ``Coleções`` section:

      * Entire library — active by default; count == sum of scenes
        across the registry.
      * Rural exteriors / Title cards / Dialogues / Night scenes — four
        curated semantic buckets. Counts are static placeholders for
        Phase 1; they become real once the (Month-2) Buscar pipeline
        produces per-bucket inverted indices.

    The dicts intentionally carry localizable labels (the templates wrap
    them in ``_()``), an icon name resolved by the icon macro, a
    pre-computed count for the ``.ct`` slot, an ``active`` flag, and a
    category token consumed by chrome.css to colour the folder glyph.
    """
    # NOTE: labels are msgids, NOT pre-translated strings — the template
    # wraps each one in `_()` so the per-request locale is honoured.
    return [
        {
            "active": True,
            "label": "Entire library",
            "icon": "grid",
            "count": total_scenes,
            "category": None,
            "url": "/scenes",
        },
        {
            "active": False,
            "label": "Rural exteriors",
            "icon": "folder",
            "count": 142,
            "category": "exterior",
            "url": "/scenes?bucket=exterior",
        },
        {
            "active": False,
            "label": "Title cards",
            "icon": "folder",
            "count": 28,
            "category": "cartela",
            "url": "/scenes?bucket=cartela",
        },
        {
            "active": False,
            "label": "Dialogues",
            "icon": "folder",
            "count": 96,
            "category": "dialogo",
            "url": "/scenes?bucket=dialogo",
        },
        {
            "active": False,
            "label": "Night scenes",
            "icon": "folder",
            "count": 73,
            "category": "interior",
            "url": "/scenes?bucket=interior",
        },
    ]


def build_chrome_context(cfg: Any, current_slug: str | None = None) -> ChromeContext:
    """Return the chrome bag merged into every full-page template context.

    The result is a flat dict ready to be ``**`` -unpacked into
    :func:`api.deps.make_ctx`. Keys mirror what the chrome partials
    expect; defaults stay in ``make_ctx`` so routes that DON'T flow
    through ``render_page`` still render the shell sensibly.

    Args:
      cfg: Loaded config namespace returned by ``cinemateca.config.load_config``.
        Typed as ``Any`` to match the existing service-layer convention
        (see ``search_service``, ``film_context``, ``catalog``); the only
        attribute touched here is ``cfg.paths.library_dir``.
      current_slug: Slug of the currently-selected film (``?film=<slug>``),
        used to mark the matching ``.ch-film`` row ``.active``.

    Returns:
      A :class:`ChromeContext` ``TypedDict`` with:
        * ``films``: full registry-backed film list (`list[Film]`).
        * ``library_state``: aggregate ``LibraryState`` (raw_present,
          index_present, scene_count, is_processed).
        * ``total_runtime_minutes``: sum of film runtimes in MINUTES.
          ``Film`` does not (yet) carry a runtime field, so this is
          always ``0`` in Phase 1. The footer renders ``0h 00m``.
        * ``current_slug``: echoed for downstream templates.
        * ``active_job_slugs``: ``set[str]`` of slugs currently processing.
          Today the registry stores video_paths, not slugs — derive the
          slug by basename-stem match with the films list.
        * ``active_job_count``: integer for the IconRail / TopBar badge.
        * ``film_match_pct``: ``dict[slug, int 0..100]`` for the per-row
          progress bar. Empty in Phase 1 (search overlays it).
        * ``film_match_counts``: ``dict[slug, int]`` shown in the row's
          ``.m`` slot. Empty in Phase 1.
        * ``collections``: list of dicts (see :func:`_default_collections`).
        * ``viewers``: list of viewer-stack dicts. Empty in Phase 1.
        * ``current_user``: identity dict or ``None``. ``None`` in Phase 1.
        * ``notification_count``: bell red-dot counter. ``0`` in Phase 1.
    """
    from cinemateca.library import library_state

    library_dir = Path(cfg.paths.library_dir)
    films = list_films(library_dir)
    lstate = library_state(library_dir)

    # ── Active job slug derivation ────────────────────────────────────
    # ``JobState.video_path`` carries the source-video path; the slug is
    # implied by ``library_dir/<slug>/raw/<filename>``. Match by:
    # ``raw_path == video_path`` first (exact, multi-film safe); fall
    # back to bare-filename match (legacy single-film layouts where the
    # job records the raw filename only).
    jobs = active_jobs()
    active_slugs: set[str] = set()
    for job in jobs:
        jp = Path(job.video_path).resolve() if job.video_path else None
        jp_name = jp.name if jp else None
        for f in films:
            try:
                if jp is not None and f.raw_path.resolve() == jp:
                    active_slugs.add(f.slug)
                    break
                if jp_name and f.raw_path.name == jp_name:
                    active_slugs.add(f.slug)
                    break
            except (OSError, RuntimeError):
                # ``resolve()`` can fail on non-existent paths under some
                # filesystems; fall back to bare-name comparison.
                if jp_name and f.raw_path.name == jp_name:
                    active_slugs.add(f.slug)
                    break

    # ── Total runtime ─────────────────────────────────────────────────
    # ``Film`` does not (yet) carry a runtime — surface 0 so the footer
    # renders sensibly; the value becomes real when the Phase-5 ingest
    # captures container duration into the registry.
    total_runtime_minutes = sum(getattr(f, "runtime_minutes", 0) or 0 for f in films)

    collections = _default_collections(films, lstate.scene_count)

    return {
        "films": films,
        "library_state": lstate,
        "total_runtime_minutes": total_runtime_minutes,
        "current_slug": current_slug,
        "active_job_slugs": active_slugs,
        "active_job_count": len(jobs),
        "film_match_pct": {},
        "film_match_counts": {},
        "collections": collections,
        "viewers": [],
        "current_user": None,
        "notification_count": 0,
    }
