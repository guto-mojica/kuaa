"""Scene-list construction + scene-context resolution for the Annotate tab.

Pure data-access + business logic (no Jinja, no HTTP). The service-layer
template builders in api/services/annotations.py call these to compose
the template context.

The underscore-prefixed helpers from the service (_resolve_selected_film,
_scene_list_with_fallback) lose the underscore here — they become public
helpers of the package because the service file is no longer the sole
caller; M3+ rerank work may want them too.
"""
from __future__ import annotations

import logging
from pathlib import Path
from cinemateca.annotations.io import load_annotations
from cinemateca.library import FilmContext, derive_fps, keyframe_url, load_json, to_smpte

logger = logging.getLogger(__name__)

# Placeholder string Moondream emits when the prompt failed to produce a
# real description; such "descriptions" do not count as a valid LLM
# description for the no_llm filter. Verbatim from the pre-extraction
# annotate route (``_BROKEN_LLM``).
_BROKEN_LLM = "One or two sentences about subject"


# ── Demo collaboration thread (v1.0 launch prep, pre-backend) ─────────────────
#
# A deterministic 2-entry curator+viewer thread so the Anotar / Buscar inspector
# matches the prototype screenshots before the real comment backend ships.
# Rendering is gated on ``cfg.collaboration.demo_threads_enabled`` (template
# side), so flipping the flag off restores the pre-demo empty state with no
# code changes.



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


def scene_context(
    ctx: FilmContext,
    scenes: list,
    scene_id: int | None,
    desc_by_scene: dict,
    annotations: dict,
    *,
    demo_threads_enabled: bool = False,
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
    if demo_threads_enabled:
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


def resolve_selected_film(ctx: FilmContext):
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


def scene_list_with_fallback(
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
