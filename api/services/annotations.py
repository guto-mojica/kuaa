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

from api.services.catalog import keyframe_url, load_json
from api.services.film_context import FilmContext
from cinemateca.annotator import load as _annotator_load
from cinemateca.annotator import save as _annotator_save

logger = logging.getLogger(__name__)

# Placeholder string Moondream emits when the prompt failed to produce a
# real description; such "descriptions" do not count as a valid LLM
# description for the no_llm filter. Verbatim from the pre-extraction
# annotate route (``_BROKEN_LLM``).
_BROKEN_LLM = "One or two sentences about subject"


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
    return [
        t.strip().lower().replace(" ", "-")
        for t in raw.split(",")
        if t.strip()
    ]


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
    fd, tmp_name = tempfile.mkstemp(
        prefix=".scene_descriptions.", suffix=".tmp", dir=path.parent
    )
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

def build_scene_list(
    ctx: FilmContext, filter_mode: str
) -> tuple[list, dict, dict]:
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
) -> dict:
    """Build the template context for the annotate scene panel.

    Verbatim port of the route's ``_scene_context``. Defaults to the
    first scene when ``scene_id`` is ``None`` or not present in
    ``scenes``. ``annotated_count`` counts scenes whose (str) id is a
    key in ``annotations`` — i.e. scenes that have ANY manual tag,
    independent of LLM-description state (see the Phase-3b report's note
    on the ``no_llm`` vs ``annotated`` ambiguity; semantics preserved
    exactly, NOT changed here).
    """
    if not scenes:
        return {"scene": None, "scene_list": [], "total": 0, "annotated_count": 0}

    # Default to first scene if scene_id not found.
    if scene_id is None or not any(s["scene_id"] == scene_id for s in scenes):
        scene_id = scenes[0]["scene_id"]

    idx = next(i for i, s in enumerate(scenes) if s["scene_id"] == scene_id)
    scene = scenes[idx]

    fp = Path(scene.get("filepath", ""))
    start_s = float(scene.get("start_time_s", 0))
    end_s = float(scene.get("end_time_s", 0))

    llm = desc_by_scene.get(scene_id)
    has_llm = bool(llm and _BROKEN_LLM not in llm.get("description", ""))

    existing_tags = annotations.get(str(scene_id), [])
    annotated_count = sum(1 for s in scenes if str(s["scene_id"]) in annotations)

    return {
        "scene": scene,
        "scene_id": scene_id,
        "img_url": keyframe_url(fp, ctx.data_dir),
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
        "scene_list": scenes,
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
    """
    no_data = not bool(load_json(ctx.metadata_dir / "keyframes_metadata.json"))
    scenes, desc_by_scene, annotations = build_scene_list(ctx, filter_mode)
    all_done = (not no_data) and (not scenes) and filter_mode == "no_llm"

    # When all scenes have LLM descriptions the no_llm filter returns nothing.
    # Fall back to showing all scenes so the user can still add manual tags —
    # all_done is kept True so the template can display the informational notice.
    if all_done:
        scenes, desc_by_scene, annotations = build_scene_list(ctx, "all")
        filter_mode = "all"

    panel = scene_context(ctx, scenes, scene_id, desc_by_scene, annotations)

    return {
        "filter": filter_mode,
        "no_data": no_data,
        "all_done": all_done,
        **panel,
    }


def build_scene_panel(
    ctx: FilmContext, scene_id: int | None, filter_mode: str
) -> dict:
    """Build the scene-panel context for the ``/api/annotate/scene`` route.

    Convenience composition of :func:`build_scene_list` +
    :func:`scene_context` so the route stays a thin parse+render. Same
    keys ``annotate_scene.html`` already consumes.
    """
    scenes, desc_by_scene, annotations = build_scene_list(ctx, filter_mode)
    return scene_context(ctx, scenes, scene_id, desc_by_scene, annotations)
