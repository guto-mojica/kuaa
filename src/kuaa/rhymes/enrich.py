"""Rhyme enrichment — decorate raw Rhyme dataclass instances with the
human-readable fields the Rimas tab template displays (description,
tags, timecode, signals like shared_tags / same_film flags).

Uses the per-scene metadata loaders from kuaa.rhymes.metadata.
"""

from __future__ import annotations

from pathlib import Path

from kuaa.config import Settings
from kuaa.library import FilmContext, keyframe_url, to_smpte
from kuaa.reproducibility import make_generator
from kuaa.rhymes.algorithm import Rhyme
from kuaa.rhymes.metadata import (
    keyframe_index,
    tags_for,
)


def enrich_rhyme(
    cfg: Settings,
    rhyme: Rhyme,
    films_by_id: dict,
    *,
    kf_cache: dict[str, tuple[dict[int, dict], float, Path] | None] | None = None,
) -> dict:
    """Convert a :class:`Rhyme` into the template's echo-card shape.

    Resolves a web-served ``keyframe_url`` and SMPTE ``timecode`` from the
    rhyme scene's entry in its film's ``keyframes_metadata.json`` — both from a
    *single* load of that file per film. When ``kf_cache`` is supplied (the
    Rimas grid build passes one shared dict per request) the parsed index is
    memoised per slug, so a grid of N echoes from the same film reads and
    parses the metadata once rather than ``2·N`` times. Films that disappeared
    between the registry walk and the call collapse to empty url/timecode so
    the template can render a placeholder card.

    M1 leaves ``reason`` empty — the M3 reranker is expected to surface
    a one-line caption explaining why the rhyme was picked; the key is
    reserved here so the template's ``{{ e.reason }}`` reads do not
    silently fall to Jinja-Undefined.
    """
    slug = rhyme.film_slug
    film = films_by_id.get(slug)
    title = getattr(film, "title", None) or slug

    if kf_cache is not None and slug in kf_cache:
        index = kf_cache[slug]
    else:
        index = keyframe_index(cfg, slug)
        if kf_cache is not None:
            kf_cache[slug] = index

    img_url = ""
    timecode = ""
    if index is not None:
        by_scene, fps, data_dir = index
        entry = by_scene.get(rhyme.scene_id)
        if entry is not None:
            img_url = keyframe_url(entry.get("filepath", ""), data_dir) or ""
            start_s = float(entry.get("start_time_s") or 0.0)
            timecode = to_smpte(start_s, fps) if start_s > 0 else ""

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
    cfg: Settings | None = None,
) -> list[dict]:
    """Synthetic 5-row similarity breakdown for the Rimas inspector card.

    Components other than visual/fused are deterministically derived from
    the (anchor, echo) scene-id pair via a seeded numpy Generator so the
    bars stay stable across reloads. Replace when the M3 multi-encoder
    reranker lands.
    """
    if anchor_data is None or selected_echo is None:
        return []

    score = float(selected_echo.get("score") or 0.0)
    global_seed = getattr(cfg, "seed", 42) if cfg is not None else 42
    rng = make_generator(
        global_seed,
        anchor_data.get("scene_id", 0),
        selected_echo.get("scene_id", 0),
    )

    def _bounded(delta_pct: int) -> float:
        # Clamp to [0.40, 0.99] so bars stay visible without saturating.
        v = score + (delta_pct / 100.0)
        return max(0.40, min(0.99, v))

    return [
        {"key": "visual", "label": "Visual · CLIP", "value": score},
        {"key": "composition", "label": "Composition", "value": _bounded(int(rng.integers(-3, 6)))},
        {"key": "semantic", "label": "Semantic", "value": _bounded(int(rng.integers(-8, 2)))},
        {
            "key": "color_luma",
            "label": "Colour · Luma",
            "value": _bounded(int(rng.integers(-9, 5))),
        },
        {"key": "fused", "label": "Fused", "value": score},
    ]


def shared_tags(cfg: Settings, anchor_data: dict | None, selected_echo: dict | None) -> list[str]:
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
