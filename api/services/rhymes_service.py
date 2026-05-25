"""Rimas Visuais (cross-film visual rhymes) — Phase-5 context builder.

Drives both the ``/tab/rimas`` full-tab partial and the
``/api/rimas/echoes`` HTMX fragment. The work split mirrors every other
Mojica tab service:

  * the route is a thin HTTP wrapper that does parameter parsing +
    template dispatch only;
  * this service walks the library, resolves the anchor scene's
    metadata, calls :func:`cinemateca.rhymes.find_rhymes` for the
    cross-film kNN, and enriches each neighbour into the shape the
    template iterates on.

Anchor selection
----------------
The ``?anchor=`` query param has the form ``"<slug>/<scene_id>"`` (e.g.
``"jeca/1"``). It is the source of truth for which scene the page is
"reading" from. When the param is absent or malformed the service falls
back to the first registered film that has at least one processed scene
on disk, with ``scene_id=1`` — Task 22's template renders an
empty-state branch when ``anchor_scene`` ends up ``None`` (e.g. the
library is empty or no film has been processed).

The service deliberately does NOT raise on unresolvable anchors. The UX
contract is "show the empty state, never crash"; the route stays
200-only for both the page and the HTMX fragment.

Future M3 swap
--------------
M3 replaces the cosine kNN with CLIP × CLAP fusion + MMR diversity +
cross-encoder rerank. The context shape exposed here (``anchor_film``,
``anchor_scene``, ``echoes``, ``k``, ``mmr_lambda``, ``threshold``) is
intended to stay stable through that swap — the M3 backend just fills
the ``echoes`` list with reranked + diversified hits and starts honoring
``mmr_lambda`` and ``threshold`` (today both are display-only knobs).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from api.services.catalog import derive_fps, keyframe_url, load_json, to_smpte
from cinemateca.library import FilmContext
from cinemateca import library
from cinemateca.rhymes import Rhyme, find_rhymes

logger = logging.getLogger(__name__)


def _parse_anchor(anchor: str | None) -> tuple[str | None, int | None]:
    """Split the ``?anchor=`` query value into ``(slug, scene_id)``.

    Accepts ``"<slug>/<scene_id>"``; anything else (missing param,
    missing slash, non-int scene_id) returns ``(None, None)`` so the
    caller falls back to the default-anchor branch.
    """
    if not anchor or "/" not in anchor:
        return None, None
    slug, scene_id_s = anchor.split("/", 1)
    if not slug:
        return None, None
    try:
        return slug, int(scene_id_s)
    except (TypeError, ValueError):
        return None, None


def _default_anchor(films: list[Any]) -> tuple[str | None, int | None]:
    """Pick the first processed film and scene 1 as the default anchor.

    "Processed" = ``is_processed`` flag from
    :func:`cinemateca.library.scan_library`, which derives from
    ``scene_count > 0``. Returns ``(None, None)`` when no film qualifies
    so the caller can render the empty-state branch.
    """
    slug = next((f.slug for f in films if getattr(f, "is_processed", False)), None)
    if slug is None:
        return None, None
    return slug, 1


def _load_scene_meta(cfg: Any, slug: str, scene_id: int) -> dict | None:
    """Return the anchor scene's metadata dict, or ``None`` if unresolvable.

    Mirrors :func:`api.services.scenes_service.build_inspector_context`'s
    on-disk lookup pattern: read ``keyframes_metadata.json`` for the
    keyframe + timecode, ``scene_descriptions.json`` for the moondream
    caption, and ``scene_tags.json`` / ``manual_annotations.json``
    (merged) for the tag list. Anything the file system cannot answer
    collapses to a sensible default (``""`` / ``[]``) so the template
    never sees ``None`` on a sub-field.

    Returns ``None`` only when the scene id itself cannot be located in
    the film's keyframe metadata — that is the signal Task 22's template
    uses to render the "anchor missing" empty state.
    """
    try:
        ctx = FilmContext.for_film(cfg, slug)
    except ValueError as exc:
        logger.info("rimas: unresolvable slug %r → empty anchor (%s)", slug, exc)
        return None

    kf_meta = load_json(ctx.metadata_dir / "keyframes_metadata.json") or []
    if not isinstance(kf_meta, list):
        return None

    entry: dict | None = None
    for e in kf_meta:
        try:
            if int(e.get("scene_id")) == scene_id:
                entry = e
                break
        except (TypeError, ValueError):
            continue
    if entry is None:
        return None

    fps = derive_fps(kf_meta)
    start_s = float(entry.get("start_time_s") or 0.0)
    end_s = float(entry.get("end_time_s") or 0.0)
    timecode = to_smpte(start_s, fps) if start_s > 0 else ""

    description = _description_for(ctx.metadata_dir, scene_id)
    tags = _tags_for(ctx.metadata_dir, scene_id)

    return {
        "scene_id": scene_id,
        "id": scene_id,
        "film_slug": slug,
        "keyframe_url": keyframe_url(entry.get("filepath", ""), ctx.data_dir) or "",
        "timecode": timecode,
        "start_s": start_s,
        "end_s": end_s,
        "title": None,
        "description": description,
        "tags": tags,
    }


def _description_for(metadata_dir: Path, scene_id: int) -> str:
    """Look up the moondream description for ``scene_id`` (``""`` if absent)."""
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
    """Return the merged (LLM + manual) tag list for ``scene_id`` (``[]`` if absent)."""
    from api.services.catalog import load_tag_index

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


def _enrich_rhyme(cfg: Any, rhyme: Rhyme, films_by_id: dict) -> dict:
    """Convert a :class:`Rhyme` into the template's echo-card shape.

    Resolves a web-served ``keyframe_url`` by looking up the rhyme
    scene's filepath in its film's ``keyframes_metadata.json`` (the
    rhyme's ``keyframe_path`` attribute is a synthetic placeholder
    derived from a slug + scene-id; the canonical URL comes from the
    real keyframe filepath on disk, mirrored through
    :func:`api.services.catalog.keyframe_url`). Films that disappeared
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
    timecode = _resolve_timecode(cfg, slug, rhyme.scene_id)

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


def _resolve_keyframe_url(cfg: Any, slug: str, scene_id: int) -> str:
    """Look up the served URL of the keyframe for ``(slug, scene_id)``.

    Reads ``keyframes_metadata.json`` for the film, finds the entry whose
    ``scene_id`` matches, and converts its ``filepath`` to a ``/media/...``
    URL via :func:`api.services.catalog.keyframe_url`. Returns ``""`` for
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


def _resolve_timecode(cfg: Any, slug: str, scene_id: int) -> str:
    """Return the SMPTE timecode of ``(slug, scene_id)``'s start, or ``""``."""
    try:
        ctx = FilmContext.for_film(cfg, slug)
    except ValueError:
        return ""
    kf_meta = load_json(ctx.metadata_dir / "keyframes_metadata.json") or []
    if not isinstance(kf_meta, list) or not kf_meta:
        return ""
    fps = derive_fps(kf_meta)
    for entry in kf_meta:
        try:
            if int(entry.get("scene_id")) == scene_id:
                start_s = float(entry.get("start_time_s") or 0.0)
                return to_smpte(start_s, fps) if start_s > 0 else ""
        except (TypeError, ValueError):
            continue
    return ""


def _rimas_cfg(cfg: Any) -> tuple[int, float, float]:
    """Read ``cfg.rimas.{top_n,mmr_lambda,threshold}`` with sensible defaults.

    Test configs built off a minimal SimpleNamespace may omit the
    ``rimas`` section entirely. The defaults here mirror
    ``config/default.yaml`` so a missing config never collapses the
    page.
    """
    rimas = getattr(cfg, "rimas", None)
    top_n = int(getattr(rimas, "top_n", 8))
    mmr_lambda = float(getattr(rimas, "mmr_lambda", 0.5))
    threshold = float(getattr(rimas, "threshold", 0.75))
    return top_n, mmr_lambda, threshold


def _select_echo(
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


def _signals_for_pair(
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


def _shared_tags(cfg: Any, anchor_data: dict | None, selected_echo: dict | None) -> list[str]:
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
    echo_tags = _tags_for(ctx.metadata_dir, int(selected_echo["scene_id"]))
    if not echo_tags:
        return []
    return [t for t in anchor_tags if t in set(echo_tags)]


def build_rimas_context(
    cfg: Any,
    *,
    anchor: str | None,
    echo: str | None = None,
) -> dict:
    """Build the Rimas Visuais template context.

    Returned keys match what ``partials/rimas.html`` /
    ``partials/rimas_echoes.html`` / ``partials/rimas_inspector.html``
    (Task 22) consume:

      * ``anchor_film`` — the :class:`cinemateca.library.Film` carrying
        the anchor scene, or ``None`` when no anchor resolves.
      * ``anchor_scene`` — dict shape mirrored on
        :func:`api.services.scenes_service.build_inspector_context`'s
        ``selected_scene`` (``scene_id`` / ``keyframe_url`` / ``timecode``
        / ``description`` / ``tags``). ``None`` triggers the empty-state
        branch.
      * ``echoes`` — list of enriched rhyme dicts (one per cross-film
        neighbour), each carrying ``film_slug`` / ``film_title`` /
        ``scene_id`` / ``keyframe_url`` / ``score`` / ``timecode`` /
        ``reason``.
      * ``selected_echo`` — one echo dict picked out by the ``?echo=``
        query param, or ``None``. Mutated in-place to carry a ``rank``
        key (1-based grid position) so the inspector can render the
        ``#NN`` pip without re-walking the list.
      * ``selected_echo_id`` — the scene_id of the selected echo (used
        by ``rimas_echoes.html`` to add the ``.sel`` highlight class).
      * ``shared_tags`` — list[str], intersection of anchor + selected
        echo tag sets (empty when either side absent or no overlap).
      * ``k`` / ``mmr_lambda`` / ``threshold`` — the Rimas knobs surfaced
        in the template (display-only for M1; M3 honors mmr_lambda and
        threshold).

    Never raises on an unresolvable anchor / echo — the empty state is
    the contract for both the route and the HTMX fragments.
    """
    library_dir = Path(cfg.paths.library_dir)
    films = library.scan_library(library_dir)
    films_by_id = {f.slug: f for f in films}

    slug, scene_id = _parse_anchor(anchor)
    # No implicit default anchor: show the empty state when ?anchor= is absent.
    # The UX entry points are: scenes inspector "Find visual rhymes" button, or
    # the Rimas tab's own anchor-picker controls once wired.

    anchor_data = (
        _load_scene_meta(cfg, slug, scene_id) if slug is not None and scene_id is not None else None
    )

    top_n, mmr_lambda, threshold = _rimas_cfg(cfg)

    rhymes: list[Rhyme] = []
    if anchor_data is not None and slug is not None and scene_id is not None:
        rhymes = find_rhymes(
            library_dir=library_dir,
            anchor_slug=slug,
            anchor_scene_id=scene_id,
            top_n=top_n,
        )

    enriched = [_enrich_rhyme(cfg, r, films_by_id) for r in rhymes]

    # ?echo=<slug>/<scene_id> highlights one of the echo cards and
    # populates the inspector. Re-uses _parse_anchor: it accepts the
    # same shape and returns (None, None) for malformed input.
    echo_slug, echo_scene_id = _parse_anchor(echo)
    selected_echo, _rank = _select_echo(enriched, echo_slug, echo_scene_id)
    selected_echo_id = selected_echo["scene_id"] if selected_echo else None

    shared = _shared_tags(cfg, anchor_data, selected_echo)

    # Attach a per-pair similarity breakdown to selected_echo so the
    # inspector's "Por que esta rima" / "Why this rhyme" card renders the
    # full bar chart from the prototype instead of a single cosine row.
    # The values are deterministic per (anchor, echo) pair so they don't
    # flicker across reloads; until the M3 multi-encoder reranker lands,
    # the components (composition / semantic / colour) are synthesized
    # around the real CLIP cosine score — flagged in the docstring of
    # _signals_for_pair so future readers don't confuse them with real
    # model outputs.
    if selected_echo is not None and not selected_echo.get("signals"):
        selected_echo["signals"] = _signals_for_pair(anchor_data, selected_echo)
        # Lazy-load the moondream description for the selected echo and
        # surface it as `reason` so the inspector's quote block renders.
        # Loading the description for every echo would bloat the grid
        # build; we only need it once the user picks a card.
        if not selected_echo.get("reason"):
            try:
                ech_ctx = FilmContext.for_film(cfg, selected_echo["film_slug"])
                selected_echo["reason"] = _description_for(
                    ech_ctx.metadata_dir, int(selected_echo["scene_id"])
                )
            except (KeyError, ValueError):
                pass

    logger.info(
        "rimas: anchor=%s/%s films=%d echoes=%d (k=%d) selected_echo=%s",
        slug,
        scene_id,
        len(films),
        len(enriched),
        top_n,
        f"{echo_slug}/{echo_scene_id}" if selected_echo else None,
    )

    return {
        "anchor_film": films_by_id.get(slug) if slug else None,
        "anchor_scene": anchor_data,
        "echoes": enriched,
        "selected_echo": selected_echo,
        "selected_echo_id": selected_echo_id,
        "shared_tags": shared,
        "k": top_n,
        "mmr_lambda": mmr_lambda,
        "threshold": threshold,
        "library_has_scenes": any(getattr(f, "is_processed", False) for f in films),
    }
