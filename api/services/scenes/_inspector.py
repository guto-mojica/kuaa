"""Inspector context builder — right ``.b-rp`` / ``.c-rp`` pane.

Extracted verbatim from ``api/services/scenes_service.py`` (lines ~105–300)
during the A1 decomposition (WS-2 Task 2).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from api.contexts import InspectorContext
from api.services.catalog import (
    derive_fps,
    keyframe_url,
    load_json,
    load_tag_index,
    to_smpte,
)
from api.services.scenes._tipo import tipo_of
from kuaa.library import FilmContext

logger = logging.getLogger(__name__)

# Valid right-pane tabs; any other value falls back to ``activity``.
_VALID_TABS = ("activity", "annotations", "properties")


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
    from kuaa.library import scan_library

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
) -> InspectorContext | None:
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
    # properties tab. Deduplicate by scene_id — detector writes N
    # keyframes per scene; only unique ids count toward the total.
    unique_ids = list(
        dict.fromkeys(e.get("scene_id") for e in kf_meta if e.get("scene_id") is not None)
    )
    total_scenes = len(unique_ids)
    scene_index = (unique_ids.index(scene_id) + 1) if scene_id in unique_ids else 1
    for i, e in enumerate(kf_meta, start=1):
        try:
            if int(e.get("scene_id")) == scene_id:
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


# ── Inspector template selector ───────────────────────────────────────────────

_VALID_KINDS = frozenset({"buscar", "cenas"})


def resolve_inspector_template(kind: str) -> tuple[str, str]:
    """Return ``(template_name, normalized_kind)`` for the given inspector kind.

    ``kind="buscar"`` (default) → ``partials/search_inspector.html``.
    ``kind="cenas"`` → ``partials/scenes_inspector.html``.
    Any unknown value falls back to ``"buscar"`` — robustness over 400s for
    a UX-only param.
    """
    normalized = kind if kind in _VALID_KINDS else "buscar"
    template_name = (
        "partials/scenes_inspector.html"
        if normalized == "cenas"
        else "partials/search_inspector.html"
    )
    return template_name, normalized
