"""Keyframe selection for a scene that owns several keyframe rows."""

from __future__ import annotations

from pathlib import Path


def pick_keyframe_entry(
    kf_meta: list[dict], scene_id: int, kf_name: str | None = None
) -> dict | None:
    """Return the keyframe row that represents ``scene_id``, or ``None``.

    The default is the middle row in file order, the same rule the canonical
    description uses, so every screen shows one picture per scene. ``kf_name``
    selects a specific row by file name and is only compared against names
    already in the metadata, never joined into a path; an unknown name falls
    back to the middle row.
    """
    rows: list[dict] = []
    for entry in kf_meta:
        try:
            if int(entry["scene_id"]) == scene_id:
                rows.append(entry)
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
    if not rows:
        return None
    if kf_name:
        for row in rows:
            if Path(str(row.get("filepath", ""))).name == kf_name:
                return row
    return rows[len(rows) // 2]
