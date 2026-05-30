"""Non-destructive AI-tag correction layer.

Curators cannot edit ``scene_tags.json`` (model output) directly — that file
is a generated artefact and rewriting it would couple curation to the AI
pipeline and lose provenance. Instead, a curator who spots a wrong LLM tag
records a *suppression* in ``tag_overrides.json``; the merge step
(:func:`cinemateca.annotations.io.merge_tag_index`) drops the suppressed
``(scene_id, tag)`` pairs at read time. ``scene_tags.json`` is never mutated,
so a suppression is fully reversible and the underlying model output stays
auditable.

On-disk schema (``metadata/tag_overrides.json``)::

    {
      "<scene_id>": {"suppressed": ["nighttime", "boat", ...]},
      ...
    }

Scene ids are stored as strings (matching ``manual_annotations.json``); tags
are stored in the canonical hyphenated-lowercase form produced by
:func:`normalize_override_tag` so matching against the merged index is
case/space-insensitive.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from cinemateca.annotations.io import atomic_write_json

if TYPE_CHECKING:
    from cinemateca.library.context import FilmContext

logger = logging.getLogger(__name__)

OVERRIDES_FILENAME = "tag_overrides.json"


def normalize_override_tag(tag: str) -> str:
    """Canonicalise a tag for suppression matching.

    Mirrors the manual-tag normalisation applied inside
    :func:`cinemateca.annotations.io.merge_tag_index`
    (``strip().lower().replace(" ", "-")``) so a suppression entered as
    ``"Night Time"`` matches a merged key of ``"night-time"``.
    """
    return tag.strip().lower().replace(" ", "-")


def load(metadata_dir: str | Path) -> dict[str, dict[str, list[str]]]:
    """Load ``tag_overrides.json`` from ``metadata_dir``.

    Returns ``{}`` when the file is absent. The shape is
    ``{scene_id_str: {"suppressed": [tag, ...]}}``.
    """
    path = Path(metadata_dir) / OVERRIDES_FILENAME
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def save(metadata_dir: str | Path, overrides: dict[str, dict[str, list[str]]]) -> Path:
    """Persist the overrides dict atomically (crash-safe ``os.replace``)."""
    path = Path(metadata_dir) / OVERRIDES_FILENAME
    atomic_write_json(path, overrides)
    logger.info("✓ Tag overrides salvos: %s (%d cenas)", path, len(overrides))
    return path


def suppressed_for_scene(overrides: dict[str, dict[str, list[str]]], scene_id: object) -> list[str]:
    """Return the (normalised) suppressed tags recorded for ``scene_id``."""
    from cinemateca.scene_ids import scene_id_key

    entry = overrides.get(scene_id_key(scene_id)) or {}
    return [normalize_override_tag(t) for t in entry.get("suppressed", [])]


def load_overrides(ctx: FilmContext) -> dict[str, dict[str, list[str]]]:
    """``FilmContext`` convenience wrapper around :func:`load`."""
    return load(ctx.metadata_dir)


def save_overrides(ctx: FilmContext, overrides: dict[str, dict[str, list[str]]]) -> Path:
    """``FilmContext`` convenience wrapper around :func:`save`."""
    return save(ctx.metadata_dir, overrides)


def set_suppressed(
    overrides: dict[str, dict[str, list[str]]],
    scene_id: object,
    tag: str,
    *,
    suppressed: bool,
) -> dict[str, dict[str, list[str]]]:
    """Add or remove a single ``(scene_id, tag)`` suppression in-place-safe.

    Returns the same dict (mutated). When ``suppressed`` is ``True`` the
    (normalised) tag is added to the scene's suppressed list if absent; when
    ``False`` it is removed and an emptied scene entry is pruned so the file
    does not accumulate dead keys.
    """
    from cinemateca.scene_ids import scene_id_key

    sid = scene_id_key(scene_id)
    norm = normalize_override_tag(tag)
    entry = overrides.setdefault(sid, {})
    current = [normalize_override_tag(t) for t in entry.get("suppressed", [])]
    if suppressed:
        if norm not in current:
            current.append(norm)
    else:
        current = [t for t in current if t != norm]
    if current:
        entry["suppressed"] = current
    else:
        overrides.pop(sid, None)
    return overrides
