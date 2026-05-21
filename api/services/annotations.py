"""Annotations service — manual scene-tagging domain logic.

This module owns what used to live inline in ``api/routes/annotate.py``:

  * loading/saving the manual-annotations dict (``load_annotations`` /
    ``save_annotations``) — ``save_annotations`` is the persist seam and
    is crash-safe because the underlying ``cinemateca.annotator.save``
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

import json
import logging
import os
import stat
import tempfile
from pathlib import Path

from api.services.catalog import derive_fps, keyframe_url, load_json, to_smpte
from api.services.film_context import FilmContext
from cinemateca.annotator import load as _annotator_load
from cinemateca.annotator import save as _annotator_save

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


# ── Tag normalization ─────────────────────────────────────────────────────────


def normalize_tags(raw: str) -> list[str]:
    """Normalize a raw comma-separated tag string to canonical tags.

    Splits on commas, then for each fragment: strips surrounding
    whitespace, drops it entirely if empty after stripping, lowercases,
    and replaces internal spaces with hyphens (compound-tag convention,
    e.g. ``"Open Field"`` -> ``"open-field"``).

    This is the EXACT transformation the annotate save route did inline
    (``[t.strip().lower().replace(" ", "-") for t in raw.split(",")
    if t.strip()]``), centralized so every consumer normalizes
    identically. Behaviour is byte-identical: order preserved,
    duplicates NOT collapsed (the prior code did not dedupe), empty
    fragments dropped.

    Args:
        raw: User-entered tag string, e.g. ``"Rural,, Open Field ,"``.

    Returns:
        Normalized tag list, e.g. ``["rural", "open-field"]``.
    """
    return [t.strip().lower().replace(" ", "-") for t in raw.split(",") if t.strip()]


# ── Persistence (atomic via cinemateca.annotator.save) ────────────────────────


def load_annotations(ctx: FilmContext) -> dict:
    """Load the manual-annotations dict for ``ctx``.

    Thin pass-through to ``cinemateca.annotator.load`` (keyed by the
    context's ``metadata_dir``). Returns ``{}`` when the file is absent,
    matching prior behaviour.
    """
    return _annotator_load(ctx.metadata_dir)


def save_annotations(ctx: FilmContext, data: dict) -> Path:
    """Persist the manual-annotations dict for ``ctx`` atomically.

    Delegates to ``cinemateca.annotator.save``, which now writes via a
    same-directory temp file + ``os.replace`` so a crash mid-write
    cannot leave a truncated ``manual_annotations.json``. The on-disk
    JSON bytes are identical to the previous plain rewrite (same
    ``indent=2, ensure_ascii=False``); only the write mechanism is
    crash-safe. Returns the path written.
    """
    return _annotator_save(ctx.metadata_dir, data)


def save_description(ctx: FilmContext, scene_id: int, new_text: str) -> None:
    """Update (or create) the description for ``scene_id`` in ``scene_descriptions.json``.

    Finds the entry whose ``scene_id`` field matches ``scene_id`` and
    replaces its ``description`` value with ``new_text``, preserving all
    other fields (e.g. ``tags``, ``objects``). If no entry exists for
    that scene, a minimal ``{"scene_id": scene_id, "description": new_text}``
    record is appended. The write is atomic (same-dir temp + os.replace)
    with the same permissions semantics as ``cinemateca.annotator.save``.
    """
    path = ctx.metadata_dir / "scene_descriptions.json"
    records: list = load_json(path) or []

    found = False
    for rec in records:
        if rec.get("scene_id") == scene_id:
            rec["description"] = new_text
            found = True
            break
    if not found:
        records.append({"scene_id": scene_id, "description": new_text})

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".scene_descriptions.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        if path.exists():
            os.chmod(tmp_path, stat.S_IMODE(os.stat(path).st_mode))
        else:
            current = os.umask(0)
            os.umask(current)
            os.chmod(tmp_path, 0o666 & ~current)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    logger.info("Description updated for scene %s", scene_id)


# ── Scene-list / scene-context builders ───────────────────────────────────────


def build_scene_list(ctx: FilmContext, filter_mode: str) -> tuple[list, dict, dict]:
    """Return ``(scene_list, desc_by_scene, annotations)`` for the tab.

    Verbatim port of the route's ``_build_scene_list``. ``filter_mode``
    ``"no_llm"`` keeps only scenes WITHOUT a valid LLM description (a
    description is "valid" if it has no ``error`` key and does not
    contain the ``_BROKEN_LLM`` placeholder); any other value keeps all
    scenes. ``desc_by_scene`` is keyed by the raw ``scene_id`` value as
    stored in ``scene_descriptions.json`` (NOT canonicalized — annotate
    used direct int keys pre-extraction; preserved to keep the rendered
    panel byte-identical).
    """
    meta_dir = ctx.metadata_dir
    kf_meta = load_json(meta_dir / "keyframes_metadata.json") or []
    descriptions = load_json(meta_dir / "scene_descriptions.json") or []
    annotations = load_annotations(ctx)

    desc_by_scene = {d["scene_id"]: d for d in descriptions if "scene_id" in d}

    valid_desc_ids = {
        d["scene_id"]
        for d in descriptions
        if "error" not in d and _BROKEN_LLM not in d.get("description", "")
    }

    if filter_mode == "no_llm":
        scenes = [s for s in kf_meta if s["scene_id"] not in valid_desc_ids]
    else:
        scenes = list(kf_meta)

    return scenes, desc_by_scene, annotations


# ── Demo collaboration thread (v1.0 launch prep, pre-backend) ─────────────────
#
# A deterministic 2-entry curator+viewer thread so the Anotar / Buscar inspector
# matches the prototype screenshots before the real comment backend ships.
# Rendering is gated on ``cfg.collaboration.demo_threads_enabled`` (template
# side), so flipping the flag off restores the pre-demo empty state with no
# code changes.


def _demo_threads_enabled() -> bool:
    """Lazy read of ``cfg.collaboration.demo_threads_enabled``.

    Imports inside the function so test fixtures that don't initialize
    the full config (or that swap in their own narrow stub) never trip
    on an import-time dependency. Failures default to ``False`` so the
    pre-demo empty state is the safe fallback.
    """
    try:
        from api.deps import get_config  # noqa: PLC0415

        cfg = get_config()
        return bool(getattr(cfg.collaboration, "demo_threads_enabled", False))
    except Exception:
        return False


def _demo_thread(timecode: str) -> list[dict]:
    """Return a small fixed curator+viewer thread for the inspector.

    The two entries mirror the prototype's Frame.io-style Annotate comments:
    a pinned curator note with attachments + reactions, and a viewer reply.
    Content is intentionally generic so it reads plausibly across any scene
    in any film; the timecode chip is interpolated so the pin sits on the
    current keyframe instead of a hardcoded one.
    """
    return [
        {
            "author": "Rafael Gonzaga",
            "role": "curator",
            "initials": "RG",
            "badge": "pinned",
            "timecode": timecode,
            "when": "há 2h",
            "body": (
                "O diálogo nesta cena é representativo da vertente "
                '<b>"campo aberto"</b> em Mazzaropi. Anexei notas do diretor '
                "e referências visuais que apareceram na pré-pesquisa."
            ),
            "attachments": [
                {"name": "notas-jeca-tatu.docx", "kind": "doc"},
                {"name": "moodboard-retrospectiva.jpg", "kind": "image"},
            ],
            "reactions": [{"emoji": "👍", "count": 2}],
            "actions": ["reply", "resolve", "share"],
        },
        {
            "author": "Júlia Reis",
            "role": "viewer",
            "initials": "JR",
            "badge": "",
            "timecode": "",
            "when": "agora",
            "body": (
                "Concordo. Talvez vincular também à temática <b>'campo aberto'</b> "
                "que tem o mesmo enquadramento? Achei interessante para uma "
                "<i>rima visual</i>."
            ),
            "attachments": [],
            "reactions": [{"emoji": "👍", "count": 0}, {"emoji": "😊", "count": 0}],
            "actions": ["reply"],
        },
    ]


def _demo_pins(timecode: str) -> list[dict]:
    """One annotation pin overlay so the keyframe canvas isn't blank.

    Position is deterministic (28% top, 42% left) so the pin sits over a
    visually neutral region of most keyframes. The label is the order in
    the thread (1 = the curator note above).
    """
    return [{"label": "1", "top": "28%", "left": "42%", "timecode": timecode}]


def _demo_comment_popup(timecode: str) -> dict:
    """Inline comment popover that floats over the keyframe.

    The popup carries a short curator quote so the .a-stage canvas reads
    like the Frame.io reference. Coordinates anchor the bubble at roughly
    the same spot as the pin so the leader line lines up.
    """
    return {
        "top": "55%",
        "left": "50%",
        "author": "Rafael Gonzaga",
        "initials": "RG",
        "when": "há 2h",
        "timecode": timecode,
        "body": (
            "O diálogo nesta cena é representativo da vertente "
            '<b>"campo aberto"</b>. Anexei notas do diretor e referências visuais.'
        ),
    }


# Static demo data — no per-call allocation. Lists are immutable (tuples)
# at module scope so accidental mutation can't pollute future requests.
_DEMO_MARKERS: tuple[dict, ...] = (
    {"pct": 18, "kind": "pin"},
    {"pct": 42, "kind": "pin"},
    {"pct": 71, "kind": "comment"},
)
_DEMO_AVATARS: tuple[dict, ...] = (
    {"initials": "EJ", "pct": 12},
    {"initials": "JF", "pct": 47},
    {"initials": "SP", "pct": 81},
)


def scene_context(
    ctx: FilmContext,
    scenes: list,
    scene_id: int | None,
    desc_by_scene: dict,
    annotations: dict,
) -> dict:
    """Build the template context for the annotate scene panel.

    Verbatim port of the route's ``_scene_context``. Defaults to the
    first scene when ``scene_id`` is ``None`` or not present in
    ``scenes``. ``annotated_count`` counts scenes whose (str) id is a
    key in ``annotations`` — i.e. scenes that have ANY manual tag,
    independent of LLM-description state (see the Phase-3b report's note
    on the ``no_llm`` vs ``annotated`` ambiguity; semantics preserved
    exactly, NOT changed here).

    Mojica Task 18 addendum: a ``selected_scene`` dict is added with
    the keyframe URL, SMPTE timecode, a short MM:SS timecode +
    duration, a progress percentage stub (0), and empty stub lists for
    ``pins`` / ``markers`` / ``timeline_avatars`` / ``timeline_ticks``
    so the ``.a-stage`` template iterates without conditionals on the
    initial render. The collaboration backend fills these in later
    milestones. All legacy keys are preserved byte-identically.
    """
    if not scenes:
        return {
            "scene": None,
            "scene_list": [],
            "total": 0,
            "annotated_count": 0,
            "comment_count": 0,
            "selected_scene": None,
        }

    # Default to first scene if scene_id not found.
    if scene_id is None or not any(s["scene_id"] == scene_id for s in scenes):
        scene_id = scenes[0]["scene_id"]

    idx = next(i for i, s in enumerate(scenes) if s["scene_id"] == scene_id)
    scene = scenes[idx]

    fp = Path(scene.get("filepath", ""))
    start_s = float(scene.get("start_time_s", 0))
    end_s = float(scene.get("end_time_s", 0))
    duration_s = max(0.0, end_s - start_s)

    llm = desc_by_scene.get(scene_id)
    has_llm = bool(llm and _BROKEN_LLM not in llm.get("description", ""))

    existing_tags = annotations.get(str(scene_id), [])
    annotated_count = sum(1 for s in scenes if str(s["scene_id"]) in annotations)

    img_url = keyframe_url(fp, ctx.data_dir)

    # Mojica .a-stage context: SMPTE for the player TC readouts; short
    # MM:SS for the .a-tl scrubrow + commentpop chips. Stubs for the
    # collaboration overlays (pins/comments/avatars) keep the template
    # branchless on initial render — backend lands in later milestones.
    fps = derive_fps(scenes)
    tc_smpte = to_smpte(start_s, fps) if start_s > 0 else "00:00:00:00"

    def _short(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        mm = int(seconds) // 60
        ss = int(seconds) % 60
        return f"{mm:02d}:{ss:02d}"

    # Mojica Task 19: the .a-rp htabs + sub-partials reach into
    # ``selected_scene`` for ``film_slug`` (HTMX ?film= propagation on
    # tab clicks), the description text (rendered as the AI .a-com.ai
    # comment in the Comments sub-partial) and the manual tags list
    # (rendered as the tag pip count on the Annotations htab). All three
    # default to safe falsy values so the sub-partials' ``{% if %}``
    # guards collapse to empty when data is absent.
    description_text = ""
    if llm and has_llm:
        description_text = llm.get("description", "") or ""

    # Collaboration overlays — populated only when the demo-threads flag
    # is on. v1.1 collaboration epic will replace this with a real backend.
    if _demo_threads_enabled():
        demo_pins = _demo_pins(tc_smpte)
        demo_popup = _demo_comment_popup(tc_smpte)
        demo_markers: list[dict] = list(_DEMO_MARKERS)
        demo_avatars: list[dict] = list(_DEMO_AVATARS)
        demo_comments = _demo_thread(tc_smpte)
    else:
        demo_pins, demo_popup, demo_markers, demo_avatars, demo_comments = [], None, [], [], []

    selected_scene: dict = {
        "scene_id": scene_id,
        "film_slug": ctx.slug,
        "keyframe_url": img_url or "",
        "timecode": tc_smpte,
        "timecode_short": _short(start_s),
        "duration_tc": _short(duration_s),
        "start_s": start_s,
        "end_s": end_s,
        "duration_s": duration_s,
        "version": None,
        "progress_pct": 0,
        "description": description_text,
        "tags": list(existing_tags),
        "pins": demo_pins,
        "comment_popup": demo_popup,
        "markers": demo_markers,
        "timeline_avatars": demo_avatars,
        "timeline_ticks": [],
        "comments": demo_comments,
        "prev_id": scenes[idx - 1]["scene_id"] if idx > 0 else None,
        "next_id": scenes[idx + 1]["scene_id"] if idx < len(scenes) - 1 else None,
    }

    # Mojica Task 19: the .a-rp Comments htab pip counts the curator
    # thread — the AI moondream description is always row #0 when present,
    # plus any demo or real curator/viewer rows.
    comment_count = (1 if description_text else 0) + len(selected_scene["comments"])

    return {
        "scene": scene,
        "scene_id": scene_id,
        "img_url": img_url,
        "start_s": start_s,
        "end_s": end_s,
        "duration_s": end_s - start_s,
        "llm": llm if has_llm else None,
        "existing_tags": existing_tags,
        "tags_value": ", ".join(existing_tags),
        "prev_id": scenes[idx - 1]["scene_id"] if idx > 0 else None,
        "next_id": scenes[idx + 1]["scene_id"] if idx < len(scenes) - 1 else None,
        "current_idx": idx,
        "total": len(scenes),
        "annotated_count": annotated_count,
        "comment_count": comment_count,
        "scene_list": scenes,
        "selected_scene": selected_scene,
    }


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


def _resolve_selected_film(ctx: FilmContext):
    """Return the ``Film`` registered for ``ctx.slug``, or ``None``.

    Used by :func:`build_annotate_context` to populate the Mojica
    breadcrumb (``Acervo / <film title> / cena NNN``). Returns ``None``
    for the global/flat context (``ctx.slug is None``) so the template
    falls back to a placeholder rather than raising on attribute access.
    Failures in the registry lookup also collapse to ``None`` (the
    annotate route deliberately tolerates missing-registry conditions —
    the legacy single-film layout has no ``films.json``).
    """
    if ctx.slug is None:
        return None
    try:
        from cinemateca.library import load_registry

        # ``library_dir`` is two parents up from ``metadata_dir`` under
        # the per-film layout: ``<library>/<slug>/metadata`` →
        # ``<library>``. Walking the path rather than re-loading the
        # config keeps this helper self-contained.
        library_dir = ctx.metadata_dir.parent.parent
        registry = load_registry(library_dir)
        entry = registry.get(ctx.slug)
        if entry is None:
            return None
        # Minimal stub that mirrors the Film dataclass attrs the
        # template reads (title / year). A future maintainer can
        # promote this to a real ``scan_library`` lookup if more fields
        # are needed; for the breadcrumb the title is enough.
        from types import SimpleNamespace

        return SimpleNamespace(
            slug=ctx.slug,
            title=entry.get("title") or ctx.slug,
            year=entry.get("year"),
        )
    except Exception:
        return None


def _scene_list_with_fallback(
    ctx: FilmContext, filter_mode: str
) -> tuple[list, dict, dict, str, bool, bool]:
    """Filtered scene list with the ``no_llm`` → ``all`` fallback applied.

    When every scene already has a valid LLM description the ``no_llm``
    filter returns nothing while data still exists. Both annotate render
    paths must then fall back to ``filter="all"`` so the user can still
    add manual tags AND the scene panel stays renderable —
    ``annotate_scene.html`` unconditionally reads ``current_idx`` /
    ``total``, so an empty list would raise ``jinja2.UndefinedError``
    (HTTP 500). Returns the (possibly re-queried) scene data plus the
    resolved ``filter_mode``, the ``all_done`` flag, and ``no_data``.

    Shared by :func:`build_annotate_context` (the ``/tab/annotate`` path)
    and :func:`build_scene_panel` (the ``/api/annotate/scene`` HTMX-nav
    path) so the two cannot drift apart again.
    """
    no_data = not bool(load_json(ctx.metadata_dir / "keyframes_metadata.json"))
    scenes, desc_by_scene, annotations = build_scene_list(ctx, filter_mode)
    all_done = (not no_data) and (not scenes) and filter_mode == "no_llm"
    if all_done:
        scenes, desc_by_scene, annotations = build_scene_list(ctx, "all")
        filter_mode = "all"
    return scenes, desc_by_scene, annotations, filter_mode, all_done, no_data


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
