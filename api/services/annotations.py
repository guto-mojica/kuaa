"""Annotations service — manual scene-tagging domain logic.

This module owns what used to live inline in ``api/routes/annotate.py``:

  * loading/saving the manual-annotations dict (``load_annotations`` /
    ``save_annotations``) — ``save_annotations`` is the persist seam and
    is crash-safe because the underlying ``cinemateca.annotations.io.save``
    now writes atomically (same-dir temp file + ``os.replace``);
  * the route-side tag-normalization list-comp, centralized as the
    documented :func:`normalize_tags` helper so the save path and any
    future consumer share ONE byte-identical normalization;
  * the annotate scene-list / scene-context builders
    (``_build_scene_list`` / ``_scene_context``) and the annotate-tab
    context builder (``build_annotate_context``).

Shared JSON-load and keyframe-URL primitives are NOT reimplemented here
— they delegate to ``api/services/catalog.py`` (``load_json`` /
``keyframe_url``, Phase 3a). Path resolution flows through
:class:`FilmContext` for consistency with the catalog service.

Behaviour is byte-preserved relative to the pre-extraction route code:
this is a refactor, not a feature change. In particular the
"Without LLM description" (``no_llm``) filter and the ``annotated``
count semantics are reproduced exactly as the route had them (see the
note on the ``no_llm`` / ``annotated`` ambiguity in the Phase-3b
report — that is a product question, intentionally NOT changed here).
"""

from __future__ import annotations

import logging

from cinemateca.library import FilmContext
from cinemateca.annotations.io import (  # noqa: F401
    load_annotations,
    normalize_tags,
    save_annotations,
)
from cinemateca.annotations.descriptions import save_description  # noqa: F401
from cinemateca.annotations.scenes import (  # noqa: F401
    build_scene_list,
    resolve_selected_film as _resolve_selected_film,
    scene_context,
    scene_list_with_fallback as _scene_list_with_fallback,
)

logger = logging.getLogger(__name__)

# Placeholder string Moondream emits when the prompt failed to produce a
# real description; such "descriptions" do not count as a valid LLM
# description for the no_llm filter. Verbatim from the pre-extraction
# annotate route (``_BROKEN_LLM``).
_BROKEN_LLM = "One or two sentences about subject"

# Mojica Task 19: valid right-pane htab values for the .a-rp shell
# (Comments / Annotations / Properties). Any other value falls back to
# ``comments`` — same defensive contract the Buscar inspector uses for
# its ``inspector_tab`` query param.
_VALID_ANNOTATE_TABS = ("comments", "annotations", "properties")


def normalize_annotate_tab(tab: str | None) -> str:
    """Return a valid ``annotate_tab`` value or fall back to ``"comments"``.

    The Anotar right pane (.a-rp) renders three htabs — Comments,
    Annotations, Properties — selected by the ``?tab=`` query parameter
    on ``/api/annotate/scene``. Unknown / missing values collapse to
    Comments (the default landing state).
    """
    if tab in _VALID_ANNOTATE_TABS:
        return tab
    return "comments"


# ── Tab context builder ───────────────────────────────────────────────────────


def build_annotate_context(
    ctx: FilmContext,
    filter_mode: str = "no_llm",
    scene_id: int | None = None,
) -> dict:
    """Build the template context the annotate tab partial needs.

    Verbatim port of the route's ``build_annotate_context``. Shared by
    the ``/tab/annotate`` HTMX fragment and the ``/annotate`` full-page
    route so both render identical markup (including the no_data /
    all_done empty-state branches). Same keys/values the templates
    already consume.

    Mojica Task 18 addendum: a ``selected_film`` key is also emitted
    (resolved from ``ctx.slug`` when present), giving the new
    ``.a-stage`` breadcrumb a real film title. ``None`` when the
    context is global/flat (single-film legacy layout) — the template
    falls back to a placeholder string. Pre-existing keys are unchanged.
    """
    scenes, desc_by_scene, annotations, filter_mode, all_done, no_data = _scene_list_with_fallback(
        ctx, filter_mode
    )
    panel = scene_context(ctx, scenes, scene_id, desc_by_scene, annotations)

    return {
        "filter": filter_mode,
        "no_data": no_data,
        "all_done": all_done,
        "selected_film": _resolve_selected_film(ctx),
        **panel,
    }


def build_scene_panel(ctx: FilmContext, scene_id: int | None, filter_mode: str) -> dict:
    """Build the scene-panel context for the ``/api/annotate/scene`` route.

    Convenience composition of :func:`build_scene_list` +
    :func:`scene_context` so the route stays a thin parse+render. Applies
    the same ``no_llm`` → ``all`` fallback as :func:`build_annotate_context`
    (via :func:`_scene_list_with_fallback`) so the HTMX-nav endpoint never
    renders ``annotate_scene.html`` with an empty list. Same keys
    ``annotate_scene.html`` already consumes.
    """
    scenes, desc_by_scene, annotations, _filter, _all_done, _no_data = _scene_list_with_fallback(
        ctx, filter_mode
    )
    return scene_context(ctx, scenes, scene_id, desc_by_scene, annotations)
