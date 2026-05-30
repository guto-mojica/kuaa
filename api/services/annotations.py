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

Data-access and business logic live in ``cinemateca.annotations.*``
(``scenes.py``, ``io.py``). This module is a thin template-layer adapter:
it calls those pure helpers and composes the Jinja template context for
the annotate route. Path resolution flows through :class:`FilmContext`.

Behaviour is byte-preserved relative to the pre-extraction route code:
this is a refactor, not a feature change. In particular the
"Without LLM description" (``no_llm``) filter and the ``annotated``
count semantics are reproduced exactly as the route had them (see the
note on the ``no_llm`` / ``annotated`` ambiguity in the Phase-3b
report — that is a product question, intentionally NOT changed here).
"""

from __future__ import annotations

import logging

from api.deps import get_config
from cinemateca.annotations.descriptions import save_description  # noqa: F401
from cinemateca.annotations.io import (  # noqa: F401
    load_annotations,
    normalize_tags,
    save_annotations,
)
from cinemateca.annotations.scenes import (  # noqa: F401
    build_scene_list,
    scene_context,
)
from cinemateca.annotations.scenes import (
    resolve_selected_film as _resolve_selected_film,
)
from cinemateca.annotations.scenes import (
    scene_list_with_fallback as _scene_list_with_fallback,
)
from cinemateca.library import FilmContext

logger = logging.getLogger(__name__)


def resolve_film_context(
    slug: str | None,
    request=None,
) -> FilmContext:
    """Resolve a ``FilmContext`` from ``?film=<slug>`` or the request cookie.

    Centralises the ``slug → for_film / request → film_ctx / fallback``
    resolution pattern shared by every annotate route handler.  Task 10
    (A6 FilmContext dependency) will consolidate this into a FastAPI
    ``Depends`` — for now it is a plain helper so the route bodies stay
    small. Accepts ``request=None`` for call sites that always supply a
    slug.
    """
    from api.deps import film_ctx

    cfg = get_config()
    if slug is not None:
        return FilmContext.for_film(cfg, slug)
    if request is not None:
        return film_ctx(request, cfg)
    return FilmContext.from_config(cfg)

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
    cfg = get_config()
    demo_threads = bool(getattr(getattr(cfg, "collaboration", None), "demo_threads_enabled", False))
    panel = scene_context(
        ctx, scenes, scene_id, desc_by_scene, annotations, demo_threads_enabled=demo_threads
    )

    return {
        "filter": filter_mode,
        "no_data": no_data,
        "all_done": all_done,
        "selected_film": _resolve_selected_film(ctx),
        **panel,
    }


def build_description_edit_context(
    fctx: FilmContext,
    scene_id: int,
    filter: str = "no_llm",
) -> dict:
    """Build the context for the ``/api/annotate/description/edit`` route.

    Looks up the current description for ``scene_id`` from
    ``scene_descriptions.json`` and returns it as ``current_description``.
    Also returns ``scene_id`` and ``filter`` so the template can wire the
    save form correctly.
    """
    from api.services.catalog import load_json

    descriptions = load_json(fctx.metadata_dir / "scene_descriptions.json") or []
    current = next(
        (d.get("description", "") for d in descriptions if d.get("scene_id") == scene_id),
        "",
    )
    return {"scene_id": scene_id, "filter": filter, "current_description": current}


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
    cfg = get_config()
    demo_threads = bool(getattr(getattr(cfg, "collaboration", None), "demo_threads_enabled", False))
    return scene_context(
        ctx, scenes, scene_id, desc_by_scene, annotations, demo_threads_enabled=demo_threads
    )
