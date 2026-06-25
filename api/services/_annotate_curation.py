"""Per-tag curation service helpers (split from api/services/annotations.py).

The v0.8-rc annotations-as-retrieval feature (commit 0c7f12d) added per-tag
delete / rename of manual tags and non-destructive AI-tag suppression. These
mutate the annotations file (or the ``tag_overrides.json`` override layer) and
are re-exported on ``api.services.annotations`` so caller import paths are
unchanged; they live here to keep ``annotations.py`` within its LOC cap.

Data-access primitives come from ``kuaa.annotations.*`` (io + overrides);
this module is the thin curation-orchestration layer.
"""

from __future__ import annotations

import logging

from kuaa.annotations.io import load_annotations, normalize_tags, save_annotations
from kuaa.annotations.overrides import (
    load_overrides,
    normalize_override_tag,
    save_overrides,
    set_suppressed,
)
from kuaa.library import FilmContext

logger = logging.getLogger(__name__)


def dedupe_tags(tags: list[str]) -> list[str]:
    """Drop duplicate tags, preserving first-seen order.

    Applied only on the per-tag curation paths (delete / rename) — NOT in
    :func:`normalize_tags`, whose no-dedupe / order-preserving contract is
    pinned by snapshot tests and shared with the byte-preserved bulk-save
    route.
    """
    seen: set[str] = set()
    out: list[str] = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def delete_manual_tag(ctx: FilmContext, scene_id: int, tag: str) -> None:
    """Remove a single manual tag from a scene, pruning an emptied entry.

    Matching is normalisation-aware (``normalize_tags`` form) so the chip
    label round-trips regardless of how it was originally entered. A no-op
    when the scene or tag is absent.
    """
    target = normalize_override_tag(tag)
    ann = load_annotations(ctx)
    sid = str(scene_id)
    current = ann.get(sid)
    if not current:
        return
    kept = dedupe_tags([t for t in current if normalize_override_tag(t) != target])
    if kept:
        ann[sid] = kept
    else:
        ann.pop(sid, None)
    save_annotations(ctx, ann)
    logger.info("Deleted tag %r from scene %s", target, scene_id)


def rename_manual_tag(ctx: FilmContext, scene_id: int, old_tag: str, new_tag: str) -> None:
    """Rename one manual tag in place (normalised + deduped).

    The new tag is normalised via :func:`normalize_tags`; an empty result
    (e.g. the curator cleared the field) falls back to a plain delete. Order
    is preserved and duplicates collapse.
    """
    old_norm = normalize_override_tag(old_tag)
    new_norm = normalize_tags(new_tag)
    if not new_norm:
        delete_manual_tag(ctx, scene_id, old_tag)
        return
    ann = load_annotations(ctx)
    sid = str(scene_id)
    current = ann.get(sid)
    if not current:
        return
    replaced: list[str] = []
    for t in current:
        replaced.extend(new_norm if normalize_override_tag(t) == old_norm else [t])
    ann[sid] = dedupe_tags(replaced)
    save_annotations(ctx, ann)
    logger.info("Renamed tag %r -> %r on scene %s", old_norm, new_norm, scene_id)


def toggle_ai_tag(ctx: FilmContext, scene_id: int, tag: str, *, suppressed: bool) -> None:
    """Suppress or restore one AI-generated tag for a scene (override layer).

    Writes ``tag_overrides.json`` only; ``scene_tags.json`` (model output) is
    never mutated, so the action is fully reversible. The BM25 cache keys on
    this file's stamp, so the change is reflected on the next search with no
    explicit reindex call.
    """
    overrides = load_overrides(ctx)
    set_suppressed(overrides, scene_id, tag, suppressed=suppressed)
    save_overrides(ctx, overrides)
    logger.info(
        "%s AI tag %r on scene %s",
        "Suppressed" if suppressed else "Restored",
        normalize_override_tag(tag),
        scene_id,
    )
