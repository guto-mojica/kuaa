"""Chrome context builder — Mojica TopBar + LeftPane.

Phase-1 / Task 8 introduces a single source of truth for the Mojica
chrome context: the variables consumed by ``_topbar.html``,
``_left_pane.html`` and ``_left_pane_body.html``.

Before Task 8 every full-page route built its own sparse chrome bag in
``api/server.py::render_page``. The launch topbar now keeps only brand,
breadcrumb, and tool tabs; collaboration/notification identity keys remain in
the context for compatibility but are not rendered by the topbar.

What this module does NOT do:

  * It does not load real per-film match percentages — that data lives
    in the (Month-2) search/Rimas pipeline and depends on a query, so
    ``film_match_pct`` ships as an empty dict for Phase 1. Search will
    overlay its own values when it lands.
  * It does not source real viewers / a real notification feed. Those belong
    to the later collaboration epic and are not visible in launch chrome.
  * It does not persist collections — the five "default collections"
    are a curated list (the Mojica prototype's Coleções section). Bucket
    counts are computed from the same per-scene tipo classifier used by
    the Scenes tab.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, TypedDict

from api.jobs import active_jobs
from kuaa.library import Film, LibraryState, scan_library


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


def _collection_counts(cfg: Any, films: list[Film]) -> Counter[str]:
    """Count default collection buckets from per-film scene metadata.

    The left-pane collection shortcuts route into ``/scenes?bucket=<tipo>``.
    To keep the displayed counts honest, reuse the same ``tipo_of`` classifier
    that builds Cenas cards instead of maintaining separate static metadata.
    Missing or malformed per-film artefacts are skipped; the overall
    ``Entire library`` count still comes from ``LibraryState.scene_count``.
    """
    from api.services.catalog import load_metadata
    from api.services.scenes import tipo_of
    from kuaa.library import FilmContext
    from kuaa.scene_ids import scene_id_key

    counts: Counter[str] = Counter()
    for film in films:
        try:
            ctx = FilmContext.for_film(cfg, film.slug)
            kf_meta, desc_by_scene, _vis_by_scene, tag_index = load_metadata(ctx.metadata_dir)
        except (OSError, TypeError, ValueError):
            continue

        tags_by_scene: dict[str, list[str]] = {}
        for tag, scene_ids in tag_index.items():
            for scene_id in scene_ids:
                tags_by_scene.setdefault(scene_id_key(scene_id), []).append(tag)

        for entry in kf_meta:
            if not isinstance(entry, dict) or "scene_id" not in entry:
                continue
            sid = scene_id_key(entry["scene_id"])
            desc = desc_by_scene.get(sid, {}).get("description") or ""
            counts[tipo_of(tags_by_scene.get(sid, []), desc)] += 1
    return counts


def _default_collections(
    total_scenes: int, bucket_counts: Counter[str], current_bucket: str | None = None
) -> list[dict[str, Any]]:
    """Return the curated default-collection list rendered in the LeftPane.

    Five entries match the Mojica prototype's ``Coleções`` section:

      * Entire library — active by default; count == sum of scenes
        across the registry.
      * Rural exteriors / Title cards / Dialogues / Night scenes — four
        curated semantic buckets. Counts come from the per-scene tipo
        classifier shared with the Cenas grid.

    The dicts intentionally carry localizable labels (the templates wrap
    them in ``_()``), an icon name resolved by the icon macro, a
    pre-computed count for the ``.ct`` slot, an ``active`` flag, and a
    category token consumed by chrome.css to colour the folder glyph.
    """
    # NOTE: labels are msgids, NOT pre-translated strings — the template
    # wraps each one in `_()` so the per-request locale is honoured.
    return [
        {
            "active": current_bucket is None,
            "label": "Entire library",
            "icon": "grid",
            "count": total_scenes,
            "category": None,
            "url": "/scenes",
        },
        {
            "active": current_bucket == "exterior",
            "label": "Exteriors",
            "icon": "folder",
            "count": bucket_counts["exterior"],
            "category": "exterior",
            "url": "/scenes?bucket=exterior",
        },
        {
            "active": current_bucket == "cartela",
            "label": "Title cards",
            "icon": "folder",
            "count": bucket_counts["cartela"],
            "category": "cartela",
            "url": "/scenes?bucket=cartela",
        },
        {
            "active": current_bucket == "dialogo",
            "label": "Dialogues",
            "icon": "folder",
            "count": bucket_counts["dialogo"],
            "category": "dialogo",
            "url": "/scenes?bucket=dialogo",
        },
        {
            "active": current_bucket == "interior",
            "label": "Night scenes",
            "icon": "folder",
            "count": bucket_counts["interior"],
            "category": "interior",
            "url": "/scenes?bucket=interior",
        },
    ]


def build_chrome_context(
    cfg: Any, current_slug: str | None = None, current_bucket: str | None = None
) -> ChromeContext:
    """Return the chrome bag merged into every full-page template context.

    The result is a flat dict ready to be ``**`` -unpacked into
    :func:`api.deps.make_ctx`. Keys mirror what the chrome partials
    expect; defaults stay in ``make_ctx`` so routes that DON'T flow
    through ``render_page`` still render the shell sensibly.

    Args:
      cfg: Loaded config namespace returned by ``kuaa.config.load_config``.
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
        * ``viewers`` / ``current_user`` / ``notification_count``:
          compatibility keys for future collaboration chrome. Not rendered by
          the launch topbar.
    """
    from kuaa.library import library_state

    library_dir = Path(cfg.paths.library_dir)
    films = scan_library(library_dir)
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
                if jp_name and f.raw_path.name == jp_name:
                    active_slugs.add(f.slug)
                    break

    # ``Film`` does not (yet) carry a runtime — surface 0 for the footer.
    total_runtime_minutes = sum(getattr(f, "runtime_minutes", 0) or 0 for f in films)

    collections = _default_collections(
        lstate.scene_count, _collection_counts(cfg, films), current_bucket=current_bucket
    )

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
