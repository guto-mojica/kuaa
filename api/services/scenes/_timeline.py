"""Bottom timeline (``.b-tl``) context builder — Task 13.

Extracted verbatim from ``api/services/scenes_service.py`` (lines ~303–511)
during the A1 decomposition (WS-2 Task 2).
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from api.services.catalog import (
    derive_fps,
    keyframe_url,
    load_json,
    to_smpte,
)
from api.services.scenes._inspector import build_inspector_context
from cinemateca.library import FilmContext

logger = logging.getLogger(__name__)


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
