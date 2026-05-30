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
"reading" from. When the param is absent, malformed, or unresolvable, the
service returns ``anchor_scene=None`` and the template renders the empty-state
branch. There is no implicit default anchor in the current branch.

The service deliberately does NOT raise on unresolvable anchors. The UX
contract is "show the empty state, never crash"; the route stays
200-only for both the page and the HTMX fragment.

Future M3 swap
--------------
The context shape exposed here (``anchor_film``, ``anchor_scene``, ``echoes``,
``k``, ``mmr_lambda``, ``threshold``) is intended to stay stable as the Rimas
backend evolves. The current branch runs cross-film visual kNN with optional
MMR diversity. Future work may add fusion or cross-encoder rerank signals.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from cinemateca import library
from cinemateca.library import FilmContext
from cinemateca.rhymes import Rhyme, find_rhymes
from cinemateca.rhymes.anchor import (
    parse_anchor as _parse_anchor,
)
from cinemateca.rhymes.config import rimas_cfg as _rimas_cfg  # noqa: F401
from cinemateca.rhymes.enrich import (  # noqa: F401
    enrich_rhyme as _enrich_rhyme,
)
from cinemateca.rhymes.enrich import (
    select_echo as _select_echo,
)
from cinemateca.rhymes.enrich import (
    shared_tags as _shared_tags,
)
from cinemateca.rhymes.enrich import (
    signals_for_pair as _signals_for_pair,
)
from cinemateca.rhymes.metadata import (  # noqa: F401
    description_for as _description_for,
)
from cinemateca.rhymes.metadata import (
    load_scene_meta as _load_scene_meta,
)

logger = logging.getLogger(__name__)


def build_rimas_context(
    cfg: Any,
    *,
    anchor: str | None,
    echo: str | None = None,
    lambda_diversity: float | None = None,
    k_candidates: int | None = None,
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
      * ``k`` / ``mmr_lambda`` / ``k_candidates`` / ``threshold`` — the Rimas
        knobs surfaced in the template. ``mmr_lambda`` and ``k_candidates`` are
        live request/config inputs; ``threshold`` is still display-only.

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

    # Resolve MMR kwargs: explicit > cfg.retrieval.rhymes.{diversity,k_candidates}
    # > hard defaults. Task 3.3 lands the cfg block; until then the getattr-chain
    # falls through to the hard defaults (0.5 / 30) which match the plan's spec
    # defaults.
    rhymes_cfg = getattr(getattr(cfg, "retrieval", None), "rhymes", None)
    if lambda_diversity is None:
        lambda_diversity = float(getattr(rhymes_cfg, "diversity", 0.5))
    if k_candidates is None:
        k_candidates = int(getattr(rhymes_cfg, "k_candidates", 30))

    rhymes: list[Rhyme] = []
    if anchor_data is not None and slug is not None and scene_id is not None:
        rhymes = find_rhymes(
            library_dir=library_dir,
            anchor_slug=slug,
            anchor_scene_id=scene_id,
            top_n=top_n,
            lambda_diversity=lambda_diversity,
            k_candidates=k_candidates,
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
        selected_echo["signals"] = _signals_for_pair(anchor_data, selected_echo, cfg)
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
        "rimas: anchor=%s/%s films=%d echoes=%d (k=%d, lambda=%.2f, k_candidates=%d) selected_echo=%s",
        slug,
        scene_id,
        len(films),
        len(enriched),
        top_n,
        lambda_diversity,
        k_candidates,
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
        "mmr_lambda": lambda_diversity,
        "k_candidates": k_candidates,
        "threshold": threshold,
        "library_has_scenes": any(getattr(f, "is_processed", False) for f in films),
    }


def rimas_context(
    cfg,
    anchor: str | None,
    echo: str | None,
    lambda_: float | None,
    k_candidates: int | None,
) -> dict:
    """Assemble the rimas context from raw route params.

    Single call-site for the three rimas handlers (tab, echoes, inspector)
    — DRYs up the identical ``build_rimas_context(cfg, anchor=…, echo=…, …)``
    calls. ``lambda_`` is the aliased FastAPI query param name (``alias="lambda"``);
    it is forwarded as ``lambda_diversity`` to :func:`build_rimas_context`.
    """
    return build_rimas_context(
        cfg,
        anchor=anchor,
        echo=echo,
        lambda_diversity=lambda_,
        k_candidates=k_candidates,
    )
