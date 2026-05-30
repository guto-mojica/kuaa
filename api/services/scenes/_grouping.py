"""Scene grouping and sorting primitives for the Cenas grid.

Peeled from ``api/services/scenes/_cards.py`` (which was itself extracted
from ``api/services/scenes_service.py``) during the A1 decomposition (WS-2
Task 2). Contains the pure grouping/sorting logic with no I/O.

Public names: ``_VALID_GROUPS``, ``_VALID_SORTS``, ``_sort_scenes``,
``_regroup``.
"""

from __future__ import annotations

from types import SimpleNamespace

from api.services.scenes._tipo import (
    _TIPO_DISPLAY_ORDER,
    _TIPO_LABEL,
)

_VALID_GROUPS = frozenset({"film", "tipo", "none"})
_VALID_SORTS = frozenset({"timecode", "duration", "pins"})


def _sort_scenes(scenes: list[dict], sort: str) -> list[dict]:
    """Stable sort of scene dicts by the requested key.

    ``timecode`` (default) → ``start_s`` ascending. The keyframe
    extractor already emits in this order, so the sort is mostly
    a no-op but it stays explicit so a future shuffled input still
    produces a sensible grid.

    ``duration`` → longest first. Matches the curator's "show me the
    long takes" intuition; short clips drift to the bottom.

    ``pins`` → ``pin_count`` descending; ties break by ``start_s``
    so two unpinned scenes stay in their original timecode order.
    """

    if sort == "duration":
        return sorted(scenes, key=lambda s: -float(s.get("duration_s") or 0.0))
    if sort == "pins":
        return sorted(
            scenes,
            key=lambda s: (
                -int(s.get("pin_count") or 0),
                float(s.get("start_s") or 0.0),
            ),
        )
    # ``timecode`` is the default — explicit sort keeps the function
    # honest for shuffled inputs even if today's pipeline emits
    # in-order.
    return sorted(scenes, key=lambda s: float(s.get("start_s") or 0.0))


def _regroup(
    per_film: list[tuple[SimpleNamespace, list[dict]]],
    *,
    group: str,
    sort: str,
) -> list[dict]:
    """Apply ``group`` + ``sort`` to the per-film scene lists.

    Each returned group carries everything the scenes_grid template
    needs to render its heading + scenecards regardless of which
    grouping mode is active — see ``_build_groups_by_film`` for the
    contract.
    """

    if group == "none":
        # One flat group across the whole library. ``is_grouped=False``
        # tells the template to skip the .group heading bar entirely.
        # ``film`` is set to the first scene's film so backward-compat
        # downstream code that still reads ``group.film.*`` doesn't
        # blow up — but the template's per-card path now reads from
        # ``s.film.*`` instead.
        flat_scenes: list[dict] = []
        for _f, scenes in per_film:
            flat_scenes.extend(scenes)
        flat_scenes = _sort_scenes(flat_scenes, sort)
        if not flat_scenes:
            return []
        first_film = flat_scenes[0].get("film") or (per_film[0][0] if per_film else None)
        return [
            {
                "film": first_film,
                "scenes": flat_scenes,
                "match_count": len(flat_scenes),
                "is_grouped": False,
                "heading_label": "",
                "heading_sub": "",
                "heading_total": 0,
                "heading_dot": "var(--c-accent)",
            }
        ]

    if group == "tipo":
        # One group per tipo (cartela / interior / exterior / dialogo /
        # transicao). Scenes from multiple films share a group; each
        # card resolves its own film via ``s.film.*`` for the inspector
        # URL. Groups appear in ``_TIPO_DISPLAY_ORDER`` regardless of
        # which tipos actually have content — empty ones drop out.
        by_tipo: dict[str, list[dict]] = {}
        for _f, scenes in per_film:
            for s in scenes:
                tipo = str(s.get("tipo") or "transicao")
                by_tipo.setdefault(tipo, []).append(s)
        out: list[dict] = []
        seen_tipos: set[str] = set()
        ordered_tipos = list(_TIPO_DISPLAY_ORDER) + sorted(
            t for t in by_tipo if t not in _TIPO_DISPLAY_ORDER
        )
        for tipo in ordered_tipos:
            scenes = by_tipo.get(tipo) or []
            if not scenes or tipo in seen_tipos:
                continue
            seen_tipos.add(tipo)
            sorted_scenes = _sort_scenes(scenes, sort)
            # Synthetic ``film`` namespace for legacy template paths
            # that still expect ``group.film``. The fields the template
            # reads under ``group.film.*`` for the HEADING (title /
            # year / director) get tipo-specific values; per-card
            # rendering reads ``s.film.*`` instead so each card's
            # film identity stays correct.
            label = _TIPO_LABEL.get(tipo, tipo.capitalize())
            tipo_ns = SimpleNamespace(
                slug=tipo,
                title=label,
                year=None,
                director=None,
                director_last=None,
                scene_count=len(scenes),
                runtime_tc="",
                runtime_s=0.0,
            )
            out.append(
                {
                    "film": tipo_ns,
                    "scenes": sorted_scenes,
                    "match_count": len(sorted_scenes),
                    "is_grouped": True,
                    "heading_label": label,
                    "heading_sub": "",
                    "heading_total": len(sorted_scenes),
                    "heading_dot": f"var(--c-cat-{tipo})",
                }
            )
        return out

    # Default: ``group=film`` — one group per film, sorted within.
    out = []
    for film_ns, scenes in per_film:
        sorted_scenes = _sort_scenes(scenes, sort)
        sub_parts: list[str] = []
        if film_ns.year:
            sub_parts.append(str(film_ns.year))
        if film_ns.director:
            sub_parts.append(film_ns.director)
        out.append(
            {
                "film": film_ns,
                "scenes": sorted_scenes,
                "match_count": len(sorted_scenes),
                "is_grouped": True,
                "heading_label": film_ns.title,
                "heading_sub": " · ".join(sub_parts),
                "heading_total": film_ns.scene_count,
                "heading_dot": "var(--c-accent)",
            }
        )
    return out
