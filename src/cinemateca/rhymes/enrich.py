"""Rhyme enrichment — decorate raw Rhyme dataclass instances with the
human-readable fields the Rimas tab template displays (description,
tags, timecode, signals like shared_tags / same_film flags).

Uses the per-scene metadata loaders from cinemateca.rhymes.metadata.
"""
from __future__ import annotations

from typing import Any

from cinemateca.library import FilmContext, keyframe_url, load_json
from cinemateca.rhymes.algorithm import Rhyme
from cinemateca.rhymes.metadata import (
    resolve_timecode,
    tags_for,
)


def _resolve_keyframe_url(cfg: Any, slug: str, scene_id: int) -> str:
    """Look up the served URL of the keyframe for ``(slug, scene_id)``.

    Reads ``keyframes_metadata.json`` for the film, finds the entry whose
    ``scene_id`` matches, and converts its ``filepath`` to a ``/media/...``
    URL via :func:`cinemateca.library.keyframe_url`. Returns ``""`` for
    any unresolvable lookup so the template can render a placeholder.
    """
    try:
        ctx = FilmContext.for_film(cfg, slug)
    except ValueError:
        return ""
    kf_meta = load_json(ctx.metadata_dir / "keyframes_metadata.json") or []
    if not isinstance(kf_meta, list):
        return ""
    for entry in kf_meta:
        try:
            if int(entry.get("scene_id")) == scene_id:
                return keyframe_url(entry.get("filepath", ""), ctx.data_dir) or ""
        except (TypeError, ValueError):
            continue
    return ""


def enrich_rhyme(cfg: Any, rhyme: Rhyme, films_by_id: dict) -> dict:
    """Convert a :class:`Rhyme` into the template's echo-card shape.

    Resolves a web-served ``keyframe_url`` by looking up the rhyme
    scene's filepath in its film's ``keyframes_metadata.json`` (the
    rhyme's ``keyframe_path`` attribute is a synthetic placeholder
    derived from a slug + scene-id; the canonical URL comes from the
    real keyframe filepath on disk, mirrored through
    :func:`cinemateca.library.keyframe_url`). Films that disappeared
    between the registry walk and the call collapse to an empty URL
    so the template can render a placeholder card.

    M1 leaves ``reason`` empty — the M3 reranker is expected to surface
    a one-line caption explaining why the rhyme was picked; the key is
    reserved here so the template's ``{{ e.reason }}`` reads do not
    silently fall to Jinja-Undefined.
    """
    slug = rhyme.film_slug
    film = films_by_id.get(slug)
    title = getattr(film, "title", None) or slug

    img_url = _resolve_keyframe_url(cfg, slug, rhyme.scene_id)
    timecode = resolve_timecode(cfg, slug, rhyme.scene_id)

    return {
        "film_slug": slug,
        "film_title": title,
        "scene_id": rhyme.scene_id,
        "id": rhyme.scene_id,
        "keyframe_url": img_url,
        "score": float(rhyme.score),
        "timecode": timecode,
        # Reserved for the M3 reranker's per-hit explanation.
        "reason": "",
    }


def select_echo(
    enriched: list[dict],
    echo_slug: str | None,
    echo_scene_id: int | None,
) -> tuple[dict | None, int | None]:
    """Resolve the ``?echo=<slug>/<scene_id>`` query param to a card.

    Walks the already-enriched echoes list (so the inspector's keyframe
    URL + score data line up byte-for-byte with the grid card it came
    from) and returns ``(echo_dict, rank)`` where ``rank`` is the 1-
    based position the card occupies in the grid. The dict is mutated
    to carry that ``rank`` so the inspector template can render the
    ``#NN`` pip without re-walking the list.

    Returns ``(None, None)`` for any unresolvable echo (the inspector
    template treats this as "anchor-only" mode and skips the .r-pair).
    """
    if echo_slug is None or echo_scene_id is None:
        return None, None
    for idx, e in enumerate(enriched, start=1):
        if e.get("film_slug") == echo_slug and int(e.get("scene_id", -1)) == echo_scene_id:
            e["rank"] = idx
            return e, idx
    return None, None


def signals_for_pair(
    anchor_data: dict | None,
    selected_echo: dict | None,
) -> list[dict]:
    """Synthetic 5-row similarity breakdown for the Rimas inspector card.

    Components other than visual/fused are deterministically derived from
    the (anchor, echo) scene-id pair so the bars stay stable across
    reloads. Replace when the M3 multi-encoder reranker lands.
    """
    if anchor_data is None or selected_echo is None:
        return []

    score = float(selected_echo.get("score") or 0.0)
    seed_key = f"{anchor_data.get('scene_id', 0)}|{selected_echo.get('scene_id', 0)}"
    seed = sum(ord(c) for c in seed_key) % 100

    def _bounded(delta_pct: int) -> float:
        # Clamp to [0.40, 0.99] so bars stay visible without saturating.
        v = score + (delta_pct / 100.0)
        return max(0.40, min(0.99, v))

    return [
        {"key": "visual", "label": "Visual · CLIP", "value": score},
        {"key": "composition", "label": "Composition", "value": _bounded((seed % 7) - 1)},
        {"key": "semantic", "label": "Semantic", "value": _bounded(-((seed + 4) % 9) + 1)},
        {"key": "color_luma", "label": "Colour · Luma", "value": _bounded(-((seed + 2) % 11) + 2)},
        {"key": "fused", "label": "Fused", "value": score},
    ]


def shared_tags(cfg: Any, anchor_data: dict | None, selected_echo: dict | None) -> list[str]:
    """Return the intersection of anchor + selected-echo tag sets.

    Used by the inspector's "Shared tags" block. Empty when either side
    is missing or no tags overlap.
    """
    if anchor_data is None or selected_echo is None:
        return []
    anchor_tags = anchor_data.get("tags") or []
    if not anchor_tags:
        return []
    try:
        ctx = FilmContext.for_film(cfg, selected_echo["film_slug"])
    except (KeyError, ValueError):
        return []
    echo_tags = tags_for(ctx.metadata_dir, int(selected_echo["scene_id"]))
    if not echo_tags:
        return []
    return [t for t in anchor_tags if t in set(echo_tags)]
