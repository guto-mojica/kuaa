"""Scenes inspector service — right ``.b-rp`` pane context builder.

The Mojica Buscar redesign (Phase 2, Task 12) ships a right-hand
inspector that swaps via HTMX from any result card's click handler:

    GET /api/scenes/{scene_id}/inspector?film=<slug>[&tab=<tab>]

The endpoint must render ``partials/search_inspector.html`` with a
self-contained context — keyframe URL, timecode, film attribution,
description, tags, plus the active tab's body. Building that context
inline in ``api/routes/scenes.py`` would duplicate the description /
tag loading already centralised in
:func:`api.routes.search._enrich_with_film_metadata`, so this service
mirrors the same lookups (description from ``scene_descriptions.json``,
tag list from the merged tag-index, SMPTE timecode from
``keyframes_metadata.json``) and exposes them through one entry point
the route consumes verbatim.

The service deliberately keeps the surface narrow:

  * the right pane is rendered for a *known* (film_slug, scene_id) pair
    only — aggregate hits already carry the slug;
  * unknown films / unknown scene_ids degrade to ``None`` so the route
    can return 404 with no template work;
  * collaboration data (comment threads, pin coords) is absent until the
    backend lands — the template renders the moondream description as a
    read-only ``.b-com.ai`` row in the Activity tab regardless;
  * signals data (``.b-sigs``) is also absent until M2 (the ``signals``
    key stays ``None`` so the template's ``{% if cfg.search.signals_enabled
    and selected_scene.signals %}`` guard hides the section);
  * rhymes data (cross-film kNN, ``.b-rimas``) is empty until Phase 5
    of the Mojica plan wires the stub service — the key is reserved
    here so callers don't break when it lands.

Tabs supported: ``activity`` (default), ``annotations``, ``properties``.
Unknown values fall back to ``activity``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from api.services.catalog import (
    build_cards,
    derive_fps,
    keyframe_url,
    load_json,
    load_metadata,
    load_tag_index,
    to_smpte,
)
from api.services.film_context import FilmContext

logger = logging.getLogger(__name__)

# Valid right-pane tabs; any other value falls back to ``activity``.
_VALID_TABS = ("activity", "annotations", "properties")

# Allowed ``tipo`` values — paired with the ``--c-cat-<tipo>`` CSS
# variables in ``web/static/css/main.css`` that colour the scene-card
# pill background. Keep in sync with that CSS block.
_TIPOS = ("cartela", "dialogo", "exterior", "interior", "transicao")


def tipo_of(tags: list[str], description: str | None) -> str:
    """Classify a scene into one of the Mojica tipo buckets.

    The Cenas-tab scene-card pill is coloured by ``--c-cat-<tipo>``;
    this classifier picks the bucket from the LLM/manual tag list and
    (as a soft fallback) the moondream description. Order matters —
    earlier branches win when a scene has tags that match multiple
    rules:

      1. ``cartela`` — opening/closing title cards. Any of the
         ``cartela`` / ``title-card`` / ``white-writing`` tags, or the
         word "title" in the description.
      2. ``interior`` — any ``interior`` or ``baixa-luz`` tag.
      3. ``exterior`` — exact ``exterior`` tag (LLM canonical) or any
         ``rural``-prefixed tag.
      4. ``dialogo`` — two-person framing / dialogue tags.
      5. ``transicao`` — default for everything else.

    Used by ``_card_to_scene`` when assembling a ``groups_by_film``
    entry for the Cenas grid template. The values pair with the
    ``--c-cat-<tipo>`` CSS variables; new values added here MUST land
    a matching CSS variable in ``main.css`` first.
    """
    desc = (description or "").lower()
    if "title" in desc or any(
        "white-writing" in t or "cartela" in t or "title-card" in t for t in tags
    ):
        return "cartela"
    if any("interior" in t or "baixa-luz" in t for t in tags):
        return "interior"
    if "exterior" in tags or any("rural" in t for t in tags):
        return "exterior"
    if any("duas-pessoas" in t or "dialogo" in t for t in tags):
        return "dialogo"
    return "transicao"


def _resolve_tab(tab: str | None) -> str:
    """Normalise the ``?tab=`` query value to a known tab key."""
    if tab in _VALID_TABS:
        return tab
    return "activity"


def _scene_lookup(kf_meta: list, scene_id: int) -> dict | None:
    """Return the keyframes_metadata entry whose ``scene_id`` matches.

    ``keyframes_metadata.json`` uses INT scene ids (as the keyframe
    extractor emits them); the inspector endpoint accepts an INT from
    the URL path, so equality is direct here.
    """
    for entry in kf_meta:
        try:
            if int(entry.get("scene_id")) == scene_id:
                return entry
        except (TypeError, ValueError):
            continue
    return None


def _films_by_slug(cfg: Any) -> dict:
    """Return ``{slug: Film}`` for every registered film.

    Local copy of ``api.services.search.films_by_id_lookup`` semantics —
    duplicated here to avoid a back-reference from the inspector service
    into the search service (which would couple two layers that should
    stay independent).
    """
    from cinemateca.library import scan_library

    library_dir = Path(cfg.paths.library_dir)
    return {film.slug: film for film in scan_library(library_dir)}


def _description_for(metadata_dir: Path, scene_id: int) -> str:
    """Look up the moondream description for ``scene_id`` (empty string if absent)."""
    descs = load_json(metadata_dir / "scene_descriptions.json") or []
    if not isinstance(descs, list):
        return ""
    for entry in descs:
        sid = entry.get("scene_id")
        if sid is None:
            continue
        try:
            if int(sid) == scene_id:
                return str(entry.get("description") or "")
        except (TypeError, ValueError):
            continue
    return ""


def _tags_for(metadata_dir: Path, scene_id: int) -> list[str]:
    """Return the merged (LLM + manual) tag list for ``scene_id``.

    Inverts the ``{tag: [scene_id, …]}`` index returned by
    ``load_tag_index``. Matches the search service's ``_enrich_with_film_metadata``
    conversion so a card and its inspector render the same tag set.
    """
    merged = load_tag_index(metadata_dir) or {}
    tags: list[str] = []
    for tag, sids in merged.items():
        if not isinstance(sids, (list, set, tuple)):
            continue
        for sid in sids:
            try:
                if int(sid) == scene_id:
                    tags.append(tag)
                    break
            except (TypeError, ValueError):
                continue
    return tags


def build_inspector_context(
    cfg: Any,
    *,
    scene_id: int,
    slug: str | None,
    inspector_tab: str = "activity",
) -> dict | None:
    """Build the template context for the right-pane inspector partial.

    Returns ``None`` when the (slug, scene_id) pair cannot be resolved —
    the route turns that into a 404 instead of rendering a blank pane.
    A non-existent film, an unknown scene id, or a per-film context with
    no on-disk metadata all collapse to ``None``.

    The returned dict is suitable for ``make_ctx(**ctx, cfg=cfg)``: it
    carries ``selected_scene`` (the unit the template iterates on),
    ``selected_film`` (for the attribution row), ``inspector_tab`` (the
    active tab key) and ``rhymes`` (empty list reserved for Phase 5).
    """
    tab = _resolve_tab(inspector_tab)

    if not slug:
        logger.info("inspector: no slug → 404 (scene_id=%d)", scene_id)
        return None

    try:
        ctx = FilmContext.for_film(cfg, slug)
    except ValueError as exc:
        logger.info("inspector: unresolvable slug %r → 404 (%s)", slug, exc)
        return None

    kf_meta = load_json(ctx.metadata_dir / "keyframes_metadata.json") or []
    if not isinstance(kf_meta, list):
        kf_meta = []
    entry = _scene_lookup(kf_meta, scene_id)
    if entry is None:
        logger.info("inspector: scene_id=%d not in %s → 404", scene_id, ctx.metadata_dir)
        return None

    films_by_slug = _films_by_slug(cfg)
    selected_film = films_by_slug.get(slug)
    if selected_film is None:
        # The slug resolves to a directory on disk but is not registered
        # in films.json. Render the inspector anyway with a stub film
        # title (the slug) so the user still sees their click landed.
        logger.info("inspector: slug %r resolves on disk but is not in films.json", slug)

    fps = derive_fps(kf_meta)
    start_s = float(entry.get("start_time_s") or 0.0)
    timecode = to_smpte(start_s, fps) if start_s > 0 else ""
    end_s = float(entry.get("end_time_s") or 0.0)
    duration_s = max(0.0, end_s - start_s)

    img_url = keyframe_url(entry.get("filepath", ""), ctx.data_dir)

    description = _description_for(ctx.metadata_dir, scene_id)
    tags = _tags_for(ctx.metadata_dir, scene_id)

    # scene_index is 1-based for the "Scene N / M" display in the
    # properties tab. ``total`` is the number of scenes in the film
    # (length of keyframes_metadata.json).
    total_scenes = len(kf_meta)
    scene_index = 1
    for i, e in enumerate(kf_meta, start=1):
        try:
            if int(e.get("scene_id")) == scene_id:
                scene_index = i
                break
        except (TypeError, ValueError):
            continue

    # ``tipo`` mirrors what ``_card_to_scene`` computes for the Cenas grid
    # so the selected scenecard's pill colour matches the right-pane
    # ``.tipo-pill`` and the ``.props`` "Tipo" row. The Buscar inspector
    # (``.b-rp``) currently ignores this field — keeping it on the scene
    # dict keeps the two surfaces sharing one shape.
    tipo = tipo_of(tags, description)

    selected_scene = {
        "id": scene_id,
        "scene_id": scene_id,
        "film_slug": slug,
        "keyframe_url": img_url or "",
        "timecode": timecode,
        "start_s": start_s,
        "end_s": end_s,
        "duration_s": duration_s,
        "tipo": tipo,
        "title": None,
        "description": description,
        "tags": tags,
        "pin_count": 0,
        "activity_count": 0,
        "annotation_count": 0,
        # Pin overlay coords — reserved for the pin-persistence backend.
        # ``None`` hides the .pin span in the template.
        "pin": None,
        # Per-modality signal breakdown — wired by M2 hybrid retrieval.
        # ``None`` hides the .b-sigs block regardless of the config flag.
        "signals": None,
        # "Described … ago" label — populated when the description carries
        # a generated_at timestamp. The default is empty so the template
        # renders nothing (a missing timestamp is the normal case today).
        "described_when": "",
        # Scene position helpers for the Properties tab.
        "scene_index": scene_index,
        "scene_total": total_scenes,
    }

    return {
        "selected_scene": selected_scene,
        "selected_film": selected_film,
        "inspector_tab": tab,
        # Cross-film visual rhymes — Phase 5 of the Mojica plan wires the
        # stub service. Empty list keeps the template's ``{% if rhymes %}``
        # guard skipping the .b-rimas block today.
        "rhymes": [],
    }


# ── Bottom timeline (.b-tl) context builder — Task 13 ────────────────────────


def _hhmm_from_seconds(seconds: float) -> str:
    """Format ``seconds`` as ``HH:MM`` for timeline tick labels.

    Used by :func:`_compute_timeline_ticks` and :func:`_format_runtime_tc`
    when the film's total duration is known (derived from the last
    keyframe's ``end_time_s``). The format intentionally drops the
    seconds component — tick labels in the prototype show ``00:00``,
    ``12:00``, ``24:00`` etc.
    """
    s = max(0, int(round(seconds)))
    hh = s // 3600
    mm = (s % 3600) // 60
    return f"{hh:02d}:{mm:02d}"


def _format_runtime_tc(runtime_s: float | None) -> str:
    """Return the ``HH:MM:SS`` runtime tag shown in the .b-tl ctrls row.

    Returns ``"--:--:--"`` when the runtime is unknown (no scenes loaded
    or last scene's ``end_time_s`` is zero) — matches the placeholder
    convention used elsewhere in the chrome when data is absent.
    """
    if not runtime_s or runtime_s <= 0:
        return "--:--:--"
    s = int(round(runtime_s))
    hh = s // 3600
    mm = (s % 3600) // 60
    ss = s % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def _compute_timeline_ticks(runtime_s: float | None, count: int = 8) -> list[str]:
    """Return ``count`` evenly-spaced ``HH:MM`` tick labels across the runtime.

    Returns an empty list when the runtime is unknown — the template's
    ``{% for tick in selected_film.timeline_ticks %}`` then produces no
    visible ticks (the strip still renders, just label-less).
    """
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
    """Build the ``.scrub > .seg`` payload for every scene in the film.

    One dict per scene with ``id``, ``keyframe_url``, ``timecode``,
    ``is_match`` (true when the scene id is in *match_scene_ids* — see
    the timeline-context builder for what that set carries today). The
    output order matches the on-disk scene order in
    ``keyframes_metadata.json`` (which is scene-id order).
    """
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
    """Build the bottom-timeline (``.b-tl``) context.

    Renders meaningfully only when both *slug* and *scene_id* resolve to
    a real film with on-disk keyframe metadata. Returns ``None``
    otherwise so the search page can skip the timeline entirely — the
    partial self-guards on ``selected_film`` AND the inspector context
    (``selected_scene``) lives on its own builder for HTMX swaps, but
    the timeline only ever renders on the full-page ``/search`` route
    where both are populated by this builder.

    The returned dict carries:

      * ``selected_film`` — a :class:`types.SimpleNamespace` exposing
        the :class:`cinemateca.library.Film` fields PLUS the timeline-
        only attributes (``scenes_for_timeline``, ``timeline_ticks``,
        ``runtime_tc``). The inspector partial reads ``.title``,
        ``.year``, ``.director`` and falls back gracefully when an attr
        is missing (Jinja-undefined → falsy), so overriding the
        ``Film`` dataclass with this namespace is safe for both
        partials sharing the search-page context.
      * ``selected_scene`` — the same scene dict the inspector builder
        produces, so the timecode + scene_id display in the ctrls row
        stays consistent across the two partials.
      * ``film_match_n`` — count of scenes flagged as ``is_match``.
      * ``query`` — verbatim string the timeline links propagate back
        into ``/search`` for state preservation.

    **Known simplifications (M1 scope):**

      1. ``is_match`` is only set for the currently selected scene.
         A full match set requires re-running the per-film search for
         the query, which would double the cost of a full-page nav and
         the timeline's match highlights are cosmetic. M2's hybrid
         retrieval layer will pre-compute the per-film match set and
         pass it through here.
      2. ``runtime_s`` derives from the last keyframe's ``end_time_s``;
         the ``Film`` dataclass has no runtime field. This is a tight
         lower bound (the actual video may extend a few seconds past
         the last detected scene) but for the prototype's tick-label
         density it is indistinguishable from the truth.
    """
    if not slug or scene_id is None:
        return None

    # Re-use the inspector builder to get a consistent (selected_scene,
    # selected_film) pair plus the on-disk scene listing.
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

    # Total runtime estimate from the last scene's end_time_s. Used both
    # for the .ctrls row's HH:MM:SS tag and for the evenly-spaced tick
    # labels under the scrub. ``None`` when no scene exposes a positive
    # end_time_s (degenerate metadata).
    runtime_s: float | None = None
    for entry in reversed(kf_meta):
        end = float(entry.get("end_time_s") or 0.0)
        if end > 0:
            runtime_s = end
            break

    # M1 simplification: highlight only the selected scene as a "match"
    # — the full per-film match set is M2's hybrid retrieval territory.
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

    # Compose the augmented Film view. SimpleNamespace lets the inspector
    # partial keep reading .title / .year / .director (returns Undefined
    # → falsy for missing director) while the timeline reads the new
    # attrs. Pulling from the Film dataclass when present preserves the
    # registry's year/title; falling back to slug otherwise.
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


# ── Cenas grid (.c-cp) — Task 15 ──────────────────────────────────────────────


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
    ``director_last`` fields (see ``cinemateca.library.Film``), but the
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


def _card_to_scene(card: dict) -> dict:
    """Convert a catalog ``build_cards`` dict to the grid template's scene shape.

    Adds the keys the new template reads (``id``, ``slug``, ``tipo``,
    ``pin_count``, ``version``, ``keyframe_url``) while preserving the
    original ``scene_id`` / ``timecode`` keys so any downstream code that
    still consumes the catalog shape keeps working.

    ``slug`` here is the *scene* slug shown on the card body (e.g.
    ``"scene 351"``) — distinct from the *film* slug carried by the
    enclosing group. Without a stable scene-slug source in the metadata
    today it falls back to ``f"scene {scene_id}"``; later phases can
    swap in a curated scene title (e.g. from a description's first
    clause) without touching the template.
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
        "tipo": tipo_of(
            list(card.get("all_tags") or card.get("tags") or []),
            card.get("full_description") or card.get("description") or "",
        ),
        "pin_count": int(card.get("pin_count") or 0),
        # ``version`` (V1/V2) is reserved for the multi-cut workflow
        # that lands in a later plan; the template hides the ``.ver``
        # pill when this is falsy.
        "version": card.get("version") or None,
    }


def _build_groups_by_film(
    cfg: Any,
    *,
    tags: list[str],
    keyword: str,
    slug: str | None = None,
) -> tuple[list[dict], list[Any], int, float, set[str]]:
    """Walk the library and produce the ``groups_by_film`` template payload.

    Returns a tuple ``(groups, films, total_scenes, total_runtime_s,
    all_tags)`` where:

      * ``groups`` is the ordered list of ``{"film": SimpleNamespace,
        "scenes": [scene_dict, ...], "match_count": int}`` dicts the
        template iterates on. Films with zero matching scenes after
        ``tags`` / ``keyword`` filtering are dropped (the heading would
        otherwise sit above an empty card area).
      * ``films`` is the raw ``list[Film]`` from ``scan_library``,
        kept around for the countrow's ``film_count``.
      * ``total_scenes`` counts cards across all films *post-filter*
        — it is the same number the toolrow's ``Filters`` pip + the
        countrow's ``scenes`` slot display.
      * ``total_runtime_s`` is the sum of per-film runtimes, also used
        only by the countrow.
      * ``all_tags`` is the union of normalized tag-index keys, kept
        as the legacy ``available_tags`` value for backwards-compat
        (the tag-filter UI moves to the right pane in a later task but
        the legacy filter must keep working).
    """
    from cinemateca.library import scan_library

    library_dir = Path(cfg.paths.library_dir)
    groups: list[dict] = []
    films: list[Any] = []
    total_scenes = 0
    total_runtime_s = 0.0
    all_tags: set[str] = set()

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
        scenes = [_card_to_scene(c) for c in cards]
        film_ns = _film_for_grid(film, kf_meta)
        total_scenes += len(scenes)
        total_runtime_s += film_ns.runtime_s
        groups.append(
            {
                "film": film_ns,
                "scenes": scenes,
                "match_count": len(scenes),
            }
        )

    return groups, films, total_scenes, total_runtime_s, all_tags


def build_cenas_context(
    cfg: Any,
    *,
    tags: list[str] | None = None,
    keyword: str = "",
    selected_scene_id: int | None = None,
    slug: str | None = None,
) -> dict:
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
    ) = _build_groups_by_film(cfg, tags=tags, keyword=keyword, slug=slug)

    # Flat cards list — preserves the legacy context key so older
    # template includes and tests that read ``cards`` directly do not
    # break during the transition. Pulls from each group's scenes so
    # the keyword/tag filter applied above is honoured.
    flat_cards: list[dict] = []
    for group in groups:
        slug = group["film"].slug
        for s in group["scenes"]:
            flat_cards.append({**s, "film_slug": slug})

    return {
        "groups_by_film": groups,
        "selected_scene_id": selected_scene_id,
        "total_scenes": total_scenes,
        "film_count": len(groups),
        "total_runtime_s": total_runtime_s,
        "total_runtime_str": _format_runtime_hm(total_runtime_s),
        # The keyframes-on-disk size is not summed today (would require
        # ``os.stat`` per keyframe — O(scenes) syscalls on every page
        # load). Em-dash placeholder until a cheap, cached source lands.
        "total_keyframes_size": "—",
        # Toolrow pip placeholders. The "fields" control isn't wired
        # today (the card layout is fixed) but the pip needs a number
        # for the template literal; 2 is the count of currently-visible
        # fields (name + tipo). The filter count reflects active tag
        # filters the legacy callers still pass.
        "visible_field_count": 2,
        "active_filter_count": len(tags),
        "no_data": total_scenes == 0,
        "available_tags": sorted(all_tags),
        "cards": flat_cards,
        # Echo back the active query so a re-render preserves the input
        # value on full-page navigation (and the inspector route can
        # plumb it through later if needed).
        "query": keyword,
    }
