"""Films-by-id lookup + Mojica template defaults + per-film and
aggregate search-context builders. Private to the search package.

Extracted from ``api/services/search.py`` (T8). All four helpers were
sized to fit inside ``cinemateca.search`` (no external pull-throughs
on the per-film path; the aggregate path crosses into
``api.services.film_context`` for the per-film loop, carved out in
``.importlinter``).

The ``api.services.film_context`` import for
:func:`build_search_context_aggregate` is the only cross of the
``cinemateca → api`` boundary in this module — it disappears in P2
when ``FilmContext`` migrates under ``cinemateca.library``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cinemateca.library.metadata import load_tag_index
from cinemateca.search.display import filter_degenerate_tags


def enrich_hits_with_film_metadata(
    cfg: Any, results: list[dict], *, per_film_slug: str | None = None
) -> list[dict]:
    """Decorate result dicts with ``film_slug``, ``description``, ``tags``.

    The Task-11 ``.b-card`` template reads ``r.film_slug``,
    ``r.description``, ``r.tags`` and ``r.pin_count``. Aggregate hits
    already carry ``film_slug`` (set by ``aggregate_search``); per-film
    hits don't, so we inject ``per_film_slug`` there. Description + tag
    fields are looked up from each film's metadata directory once
    (memoised by slug) so a 100-row result list reads each film's
    descriptions / tag_index at most once.

    Missing data is benign — ``description`` falls back to ``""`` and
    ``tags`` to ``[]``, both of which the template handles. ``pin_count``
    is always 0 today; the persistence layer for pins lands later in M1.

    Relocated from ``api/routes/search.py`` (T15). The lazy imports of
    ``api.services.catalog`` (``load_json`` + ``load_tag_index``) and
    ``api.services.film_context`` are the P2-cleanup carve-outs already
    documented for this module. ``load_tag_index`` is sourced from the
    catalog twin (not the cinemateca twin) to preserve byte-identical
    snapshot output on malformed ``scene_tags.json`` — the two loaders
    diverge only on that unhappy path (see ``_tag_index.py`` docstring).
    """
    from api.services.catalog import load_json, load_tag_index
    from api.services.film_context import FilmContext

    desc_cache: dict[str, dict[int, str]] = {}
    tag_cache: dict[str, dict[int, list[str]]] = {}

    def _load_film_meta(slug: str) -> tuple[dict[int, str], dict[int, list[str]]]:
        if slug in desc_cache:
            return desc_cache[slug], tag_cache[slug]
        try:
            ctx = FilmContext.for_film(cfg, slug)
        except ValueError:
            desc_cache[slug] = {}
            tag_cache[slug] = {}
            return desc_cache[slug], tag_cache[slug]
        descs_raw = load_json(ctx.metadata_dir / "scene_descriptions.json") or []
        descs: dict[int, str] = {}
        if isinstance(descs_raw, list):
            for entry in descs_raw:
                sid = entry.get("scene_id")
                if sid is None:
                    continue
                try:
                    descs[int(sid)] = str(entry.get("description") or "")
                except (TypeError, ValueError):
                    continue
        # Invert the merged tag_index into {scene_id: [tag, …]} so we
        # can pull per-scene tag pills in O(1). load_tag_index merges
        # LLM + manual; the v0.3 catalog template already trusts this
        # merged view to be the "displayable" tag set.
        merged = load_tag_index(ctx.metadata_dir) or {}
        per_scene: dict[int, list[str]] = {}
        for tag, sids in merged.items():
            if not isinstance(sids, (list, set, tuple)):
                continue
            for sid in sids:
                try:
                    key = int(sid)
                except (TypeError, ValueError):
                    continue
                per_scene.setdefault(key, []).append(tag)
        desc_cache[slug] = descs
        tag_cache[slug] = per_scene
        return descs, per_scene

    enriched: list[dict] = []
    for r in results:
        r = dict(r)  # don't mutate the caller's dicts
        slug = r.get("film_slug") or per_film_slug
        if slug and "film_slug" not in r:
            r["film_slug"] = slug
        sid_raw = r.get("scene_id")
        sid = None
        if sid_raw is not None:
            try:
                sid = int(sid_raw)
            except (TypeError, ValueError):
                sid = None
        if slug and sid is not None:
            descs, tags_by_scene = _load_film_meta(slug)
            r.setdefault("description", descs.get(sid, ""))
            r.setdefault("tags", tags_by_scene.get(sid, []))
        else:
            r.setdefault("description", "")
            r.setdefault("tags", [])
        r.setdefault("pin_count", 0)
        enriched.append(r)
    return enriched


def films_by_id_lookup(cfg: Any) -> dict:
    """Return ``{film.slug: film}`` for every registered film.

    Task 11's ``.b-card`` markup looks up ``films_by_id[r.film_slug]``
    to pull the film title + year onto each result card; the lookup is
    built here so both the per-film and aggregate routes (and the
    ``build_search_context*`` builders) populate the same shape.

    Returns an empty dict when the library directory is absent —
    consistent with :func:`cinemateca.library.scan_library`'s contract.
    Templates should treat the dict as a best-effort lookup
    (``films_by_id.get(slug)``); cards whose ``film_slug`` is missing
    still render with sensible fallbacks.
    """
    from cinemateca.library import scan_library

    library_dir = Path(cfg.paths.library_dir)
    return {film.slug: film for film in scan_library(library_dir)}


def mojica_search_defaults() -> dict:
    """Defaults the Mojica Buscar template (``partials/search.html``)
    needs whenever no actual query has been issued.

    Task 10 introduces a richer template context — query state, view
    toggle, results list, film lookup, highlighted tags — that previous
    tab-renders did not surface. These defaults let the page render
    the initial "type a query to search" empty state with no special
    casing on the template side.

    The per-modality result list is intentionally empty here. Task 11
    fills it with ``.b-card``-shaped dicts produced by the
    ``/api/search`` handlers; ``films_by_id`` is populated lazily by
    callers that have a cfg in hand (see :func:`films_by_id_lookup`).
    """
    return {
        "query": "",
        "total": 0,
        "film_count": 0,
        "latency_ms": None,
        "active_mode": "text",
        "active_view": "grid",
        "selected_scene_id": None,
        "results": [],
        "films_by_id": {},
        "highlighted_tags": set(),
    }


def build_search_context(ctx: Any, cfg: Any | None = None) -> dict:
    """Build the per-film search-tab partial context.

    Uses the RAW merged tag index (only its keys feed
    ``available_tags`` — identical to the normalised index's keys) and
    runs them through :func:`filter_degenerate_tags` so the pill grid
    stays clean even when ``scene_tags.json`` carries leaked caption
    fragments.

    Mojica-redesign keys (Task 10) live alongside ``available_tags`` so
    the rewritten template can render the empty state without forcing
    every route to populate them. The ``query`` / ``total`` /
    ``results`` defaults are overwritten by ``/api/search`` responses
    once a query fires.

    ``cfg`` is optional for back-compat: when supplied, ``films_by_id``
    is populated via :func:`films_by_id_lookup` so Task-11's ``.b-card``
    template can resolve film titles/years on hits returned by the
    same request. When omitted (legacy callers), ``films_by_id`` stays
    empty and the template falls back to safe-get behaviour.

    ``ctx`` is duck-typed (must expose ``metadata_dir``) — the same
    shape ``FilmContext`` provides.
    """
    tag_index = load_tag_index(ctx.metadata_dir)
    raw_tags = sorted(tag_index.keys()) if tag_index else []
    ctx_dict = mojica_search_defaults()
    ctx_dict["available_tags"] = filter_degenerate_tags(raw_tags)
    if cfg is not None:
        ctx_dict["films_by_id"] = films_by_id_lookup(cfg)
    return ctx_dict


def build_search_context_aggregate(cfg: Any) -> dict:
    """Build the aggregate search-tab context (union across all films).

    Mirrors ``api.services.catalog.build_scenes_context_aggregate``'s
    tag-union pattern: walks the library registry, unions every film's
    tag-index keys, filters degenerate entries, and returns the same
    ``available_tags`` key the per-film builder exposes — so the
    ``partials/search.html`` template renders identically in either
    mode.

    Mojica-redesign keys (Task 10) are merged in via
    :func:`mojica_search_defaults` so the aggregate path and per-film
    path expose the same context shape. ``films_by_id`` is populated
    here so the template's title/year lookup resolves on every card.
    """
    from api.services.film_context import FilmContext
    from cinemateca.library import scan_library

    library_dir = Path(cfg.paths.library_dir)
    all_tags: set[str] = set()
    for film in scan_library(library_dir):
        try:
            ctx = FilmContext.for_film(cfg, film.slug)
        except ValueError:
            continue
        tag_index = load_tag_index(ctx.metadata_dir)
        all_tags.update(tag_index.keys())
    ctx_dict = mojica_search_defaults()
    ctx_dict["available_tags"] = filter_degenerate_tags(sorted(all_tags))
    ctx_dict["films_by_id"] = films_by_id_lookup(cfg)
    return ctx_dict
