"""Catalog service — scene/metadata/card domain logic.

This module owns what used to be copy-pasted across ``api/routes/*``:

  * shared JSON-load and keyframe-URL primitives (``load_json``,
    ``keyframe_url``) — previously a private ``_load_json`` /
    ``_keyframe_url`` in *each* of scenes.py / annotate.py / search.py;
  * catalog metadata loading + tag-index merge/normalization
    (``load_metadata``, ``load_tag_index``);
  * scene-card construction + filtering (``build_cards``);
  * the scenes-tab context builder (``build_scenes_context``).

Scene-ID canonicalization is NOT reimplemented here — it delegates to
``cinemateca.scene_ids`` (``scene_id_key`` / ``normalize_tag_index``,
the Phase-1c helpers). Tag-index merge delegates to
``cinemateca.annotator.merge_tag_index``. The service only orchestrates.

All path resolution flows through :class:`FilmContext` instead of
scattered ``cfg.paths.*`` reads (see ``api/services/film_context.py``).

Behaviour is byte-preserved relative to the pre-extraction route code:
this is a refactor, not a feature/validation change. (Corrupt-index
validation is Phase 3c; it is intentionally absent here.)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from api.services.film_context import FilmContext
from api.services.film_service import list_films
from cinemateca.scene_ids import scene_id_key

logger = logging.getLogger(__name__)


# ── Shared primitives (consumed by scenes + Phase 3b/3c annotate/search) ──────

def load_json(path: Path) -> list | dict | None:
    """Load a JSON file, or return ``None`` if it does not exist.

    The exact prior ``_load_json`` body (scenes.py / annotate.py had
    identical copies). Kept permissive on return type because callers
    apply their own ``or []`` / ``or {}`` defaulting, matching the old
    behaviour.
    """
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def keyframe_url(filepath: str | Path, data_dir: Path) -> str | None:
    """Convert a stored keyframe filepath to a ``/media/...`` URL.

    Tries the path as-stored and relative to CWD, returning the first
    that resolves *inside* ``data_dir``; ``None`` otherwise (e.g. a path
    outside the served root). This unifies the two prior copies
    (``scenes._keyframe_url`` took a ``Path``; ``search._keyframe_url``
    took a ``str``) — accepting both keeps every call site byte-identical.
    """
    fp = Path(filepath)
    for candidate in (fp, Path.cwd() / fp):
        try:
            rel = candidate.resolve().relative_to(data_dir.resolve())
            return f"/media/{rel.as_posix()}"
        except ValueError:
            continue
    return None


# ── Metadata + tag index ──────────────────────────────────────────────────────

def load_tag_index(metadata_dir: Path) -> dict:
    """Load the RAW merged (un-normalized) inverted tag index.

    Mirrors ``search._load_tag_index`` exactly: read ``scene_tags.json``
    (LLM, INT ids) + ``manual_annotations.json`` (STR keys) and
    ``merge_tag_index`` them WITHOUT normalizing. Search depends on this
    raw shape — ``SemanticSearch.combined`` normalizes internally
    (Phase 1c), so passing a pre-normalized index would change nothing
    for filtering but would diverge from the characterized contract.
    Only the *keys* are used for ``available_tags`` (identical either
    way, since normalization never drops/renames tags).
    """
    from cinemateca.annotator import load as load_annotations
    from cinemateca.annotator import merge_tag_index

    tags_path = metadata_dir / "scene_tags.json"
    llm_tags: dict = {}
    if tags_path.exists():
        with open(tags_path, encoding="utf-8") as f:
            llm_tags = json.load(f)
    annotations = load_annotations(metadata_dir)
    return merge_tag_index(llm_tags, annotations)


def load_metadata(metadata_dir: Path) -> tuple[list, dict, dict, dict]:
    """Return ``(kf_meta, desc_by_scene, vis_by_scene, tag_index)``.

    Verbatim port of ``scenes._load_metadata``. The returned
    ``tag_index`` is NORMALIZED (``{tag: {canonical str id}}`` via
    ``normalize_tag_index``) so every downstream membership test the
    scenes path does is str-vs-str. ``desc_by_scene`` / ``vis_by_scene``
    are keyed by canonical str id (``scene_id_key``).
    """
    from cinemateca.annotator import load as load_annotations
    from cinemateca.annotator import merge_tag_index
    from cinemateca.scene_ids import normalize_tag_index

    kf_meta = load_json(metadata_dir / "keyframes_metadata.json") or []
    descriptions = load_json(metadata_dir / "scene_descriptions.json") or []
    llm_tags = load_json(metadata_dir / "scene_tags.json") or {}
    visual_data = load_json(metadata_dir / "visual_analysis.json") or []
    annotations = load_annotations(metadata_dir)

    desc_by_scene = {
        scene_id_key(d["scene_id"]): d for d in descriptions if "scene_id" in d
    }
    vis_by_scene = {
        scene_id_key(v["scene_id"]): v for v in visual_data if "scene_id" in v
    }
    # merge_tag_index yields a hybrid index with mixed int (LLM) / str
    # (manual) value types. Normalize to canonical str ids here so every
    # downstream membership test is str-vs-str.
    tag_index = normalize_tag_index(merge_tag_index(llm_tags, annotations))

    return kf_meta, desc_by_scene, vis_by_scene, tag_index


# ── SMPTE timecode utilities ──────────────────────────────────────────────────

def to_smpte(seconds: float, fps: float = 24.0) -> str:
    """Convert a time-in-seconds to SMPTE ``HH:MM:SS:FF`` notation."""
    fps_int = max(1, round(fps))
    total_frames = int(seconds * fps)
    ff = total_frames % fps_int
    rest = total_frames // fps_int
    ss = rest % 60
    mm = (rest // 60) % 60
    hh = rest // 3600
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


def derive_fps(kf_meta: list) -> float:
    """Infer original video FPS from keyframe metadata entries.

    Uses the first entry where both ``start_frame`` and ``start_time_s``
    are positive (the first scene always starts at 0 s / frame 0, so it
    is useless for derivation). Falls back to 24.0 when no suitable
    entry exists.
    """
    for entry in kf_meta:
        t = float(entry.get("start_time_s") or 0.0)
        f = int(entry.get("start_frame") or 0)
        if t > 0 and f > 0:
            return f / t
    return 24.0


# ── Scene-card construction ───────────────────────────────────────────────────

def _select_tags_by_frequency(
    scene_tags: list[str], tag_index: dict, n: int = 16
) -> list[str]:
    """Return up to n tags sampled across the global-frequency spectrum.

    Tags are sorted ascending by corpus frequency (rare → common). When
    the scene has ≤ n tags all are returned in that order. When it has
    more, n tags are picked at evenly-spaced positions from the sorted
    list so the selection spans the full diversity: from the most
    scene-specific labels (low frequency) to the most generic ones
    (high frequency), rather than taking the first n alphabetically.
    """
    by_freq = sorted(scene_tags, key=lambda t: len(tag_index.get(t, [])))
    if len(by_freq) <= n:
        return by_freq
    total = len(by_freq)
    return [by_freq[i * total // n] for i in range(n)]


def build_cards(
    kf_meta: list,
    desc_by_scene: dict,
    vis_by_scene: dict,
    tag_index: dict,
    data_dir: Path,
    selected_tags: list[str],
    keyword: str,
) -> list[dict]:
    """Filter ``kf_meta`` and build scene-card dicts for the template.

    Tag filter: intersect scene_ids across all selected tags. Keyword
    filter: search the description text blob. Tags on each card are
    selected by :func:`_select_tags_by_frequency` (up to 16, sampled
    across the frequency spectrum). ``tag_index`` is expected normalized
    (as :func:`load_metadata` returns it).
    """
    scenes = list(kf_meta)
    fps = derive_fps(kf_meta)

    # Tag filter — intersect scene_ids across all selected tags.
    # tag_index is already normalized to {tag: {canonical str id}}, so
    # the membership test is str-vs-str.
    if selected_tags and tag_index:
        valid_ids = set(tag_index.get(selected_tags[0], set()))
        for tag in selected_tags[1:]:
            valid_ids &= set(tag_index.get(tag, set()))
        scenes = [
            s for s in scenes if scene_id_key(s.get("scene_id", "")) in valid_ids
        ]

    # Keyword filter — search description text blob
    if keyword:
        kw = keyword.lower()
        filtered = []
        for s in scenes:
            sid = scene_id_key(s.get("scene_id", ""))
            desc = desc_by_scene.get(sid, {})
            blob = " ".join(str(v) for v in desc.values()).lower()
            if kw in blob:
                filtered.append(s)
        scenes = filtered

    cards = []
    for s in scenes:
        sid = scene_id_key(s.get("scene_id", ""))
        fp = Path(s.get("filepath", ""))
        img_url = keyframe_url(fp, data_dir)
        start_s = float(s.get("start_time_s") or 0.0)
        if start_s > 0:
            tc = to_smpte(start_s, fps)
        else:
            tc = s.get("timecode_start") or s.get("start_timecode", "")

        # Tags from tag_index (inverted lookup). tag_index ids are
        # already canonical str keys, so this is direct str-vs-str.
        all_scene_tags = list({tag for tag, ids in tag_index.items() if sid in ids})
        all_tags_sorted = sorted(all_scene_tags, key=lambda t: len(tag_index.get(t, [])))
        scene_tags = _select_tags_by_frequency(all_scene_tags, tag_index)

        # Visual analysis summary
        vis = vis_by_scene.get(sid, {})
        env = vis.get("environment", {})
        env_parts = [
            p for p in [env.get("location", ""), env.get("time_of_day", "")] if p
        ]
        num_people = vis.get("num_faces")

        # Description one-liner
        desc = desc_by_scene.get(sid, {})
        description = desc.get("description") or ""

        # Duration is end_time_s − start_time_s, both already in
        # ``s`` (keyframe metadata). We carry the raw seconds so the
        # Cenas grid's "Sort by Duration" can compare without parsing
        # the SMPTE string back into seconds.
        end_s = float(s.get("end_time_s") or 0.0)
        duration_s = max(0.0, end_s - start_s)
        cards.append(
            {
                "scene_id": s.get("scene_id"),
                "img_url": img_url,
                "timecode": tc,
                "start_s": start_s,
                "duration_s": duration_s,
                "tags": scene_tags,
                "all_tags": all_tags_sorted,
                "environment": " · ".join(env_parts),
                "num_people": num_people,
                "description": description[:120] if description else "",
                "full_description": description,
            }
        )

    return cards


# ── Tab context builders ──────────────────────────────────────────────────────

def build_scenes_context(ctx: FilmContext) -> dict:
    """Build the scenes-tab template context (no tag/keyword filter).

    Shared by the ``/tab/scenes`` HTMX fragment and the ``/scenes``
    full-page route so both render identical markup (including the
    empty-state hint when no keyframes exist). Same keys/values the
    template already consumes: ``cards``, ``available_tags``,
    ``no_data``.
    """
    kf_meta, desc_by_scene, vis_by_scene, tag_index = load_metadata(
        ctx.metadata_dir
    )
    available_tags = sorted(tag_index.keys())
    cards = build_cards(
        kf_meta, desc_by_scene, vis_by_scene, tag_index, ctx.data_dir, [], ""
    )
    return {
        "cards": cards,
        "available_tags": available_tags,
        "no_data": not kf_meta,
    }


def build_scenes_grid(ctx: FilmContext, tags: list[str], keyword: str) -> dict:
    """Build the filtered scenes-grid context for ``/api/scenes``.

    Same single key (``cards``) the ``scenes_grid.html`` partial
    consumes; behaviour identical to the prior inline route body.
    """
    kf_meta, desc_by_scene, vis_by_scene, tag_index = load_metadata(
        ctx.metadata_dir
    )
    cards = build_cards(
        kf_meta, desc_by_scene, vis_by_scene, tag_index, ctx.data_dir, tags, keyword
    )
    return {"cards": cards}


def build_scenes_grid_aggregate(cfg: Any, tags: list[str], keyword: str) -> dict:
    """Build the filtered scenes-grid context across ALL films.

    Filter-aware sibling of :func:`build_scenes_context_aggregate` for
    the ``/api/scenes`` grid-refresh endpoint (HTMX tag/keyword changes).

    Walks the library the same way the aggregate context builder does,
    but applies *tags* and *keyword* filters per film via
    :func:`build_cards`.  Each card is annotated with ``film_slug`` and
    ``film_title`` so the template can group by film when desired.

    Returns ``{"cards": all_cards}`` — matching the per-film
    :func:`build_scenes_grid` return shape — so the same
    ``scenes_grid.html`` partial works in both modes.
    """
    library_dir = Path(cfg.paths.library_dir)
    all_cards: list[dict] = []
    for film in list_films(library_dir):
        ctx = FilmContext.for_film(cfg, film.slug)
        kf_meta, desc_by_scene, vis_by_scene, tag_index = load_metadata(
            ctx.metadata_dir
        )
        cards = build_cards(
            kf_meta, desc_by_scene, vis_by_scene, tag_index, ctx.data_dir,
            tags, keyword,
        )
        for c in cards:
            c["film_slug"] = film.slug
            c["film_title"] = film.title
        all_cards.extend(cards)
    return {"cards": all_cards}


def build_scenes_context_aggregate(cfg: Any) -> dict:
    """Build the scenes context across ALL films in the library.

    For each film: load its per-film metadata, build cards from its
    artefacts (path math through ``FilmContext.for_film``), annotate
    each card with ``film_slug`` + ``film_title`` for the template
    grouping, and concatenate.

    Tolerates registered-but-unprocessed films (no ``metadata/`` dir):
    ``load_metadata`` returns empty containers and the film contributes
    zero cards without raising.

    ``available_tags`` is the union of per-film tag-index keys, already
    in their normalized form (matching ``build_scenes_context``'s shape).

    ``no_data`` is True iff no card was produced across all films. This
    diverges intentionally from ``build_scenes_context``'s ``not kf_meta``
    test: at the aggregate level "no data" means the whole library has
    nothing renderable, not that any one film lacks keyframes. Callers
    in T9 must surface this distinction in copy.

    Performance: loads all per-film metadata from disk on every call.
    For large libraries (~100+ films) consider a request-scoped cache
    alongside the per-film search index cache T8 introduces.
    """
    library_dir = Path(cfg.paths.library_dir)
    all_cards: list[dict] = []
    all_tags: set[str] = set()
    for film in list_films(library_dir):
        ctx = FilmContext.for_film(cfg, film.slug)
        kf_meta, desc_by_scene, vis_by_scene, tag_index = load_metadata(
            ctx.metadata_dir
        )
        cards = build_cards(
            kf_meta, desc_by_scene, vis_by_scene, tag_index, ctx.data_dir, [], ""
        )
        for c in cards:
            c["film_slug"] = film.slug
            c["film_title"] = film.title
        all_cards.extend(cards)
        all_tags.update(tag_index.keys())
    return {
        "cards": all_cards,
        "available_tags": sorted(all_tags),
        "no_data": not all_cards,
    }
