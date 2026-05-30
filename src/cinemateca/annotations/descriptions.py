"""Per-scene description edits — read/write scene_descriptions.json.

Separate from io.py because descriptions are a different artifact than
manual annotations (tags), even though both live in the metadata_dir.
"""

from __future__ import annotations

import json
import logging
import os
import stat
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cinemateca.library import load_json

if TYPE_CHECKING:
    from cinemateca.library.context import FilmContext

logger = logging.getLogger(__name__)


def save_description(ctx: FilmContext, scene_id: int, new_text: str) -> None:
    """Update (or create) the description for ``scene_id`` in ``scene_descriptions.json``.

    Finds the entry whose ``scene_id`` field matches ``scene_id`` and
    replaces its ``description`` value with ``new_text``, preserving all
    other fields (e.g. ``tags``, ``objects``). If no entry exists for
    that scene, a minimal ``{"scene_id": scene_id, "description": new_text}``
    record is appended. The write is atomic (same-dir temp + os.replace)
    with the same permissions semantics as ``cinemateca.annotations.io.save``.
    """
    path = ctx.metadata_dir / "scene_descriptions.json"
    raw = load_json(path)
    records: list[Any] = raw if isinstance(raw, list) else []

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
