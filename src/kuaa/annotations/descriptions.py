"""Per-scene description edits — read/write scene_descriptions.json.

Separate from io.py because descriptions are a different artifact than
manual annotations (tags), even though both live in the metadata_dir.
"""

from __future__ import annotations

import difflib
import json
import logging
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kuaa.library import load_json
from kuaa.scene_ids import scene_id_key

if TYPE_CHECKING:
    from kuaa.library.context import FilmContext

logger = logging.getLogger(__name__)

_KF_POS_RE = re.compile(r"_kf_(\d+)$")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_SIMILARITY_THRESHOLD = 0.7


def _kf_pos(record: dict, fallback: int) -> int:
    """Numeric keyframe position parsed from ``keyframe_id`` (``scene_NNNN_kf_KK``).

    Falls back to the record's original list position when ``keyframe_id`` is
    missing/unparseable (the describer's error-path rows omit it), so
    :func:`canonical_description` degrades gracefully instead of crashing.
    """
    match = _KF_POS_RE.search(str(record.get("keyframe_id", "")))
    return int(match.group(1)) if match else fallback


def _split_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def _is_novel(sentence: str, existing_normalized: list[str]) -> bool:
    norm = " ".join(sentence.lower().split())
    return not any(
        difflib.SequenceMatcher(None, norm, other).ratio() >= _SIMILARITY_THRESHOLD
        for other in existing_normalized
    )


def canonical_description(descriptions: list[dict]) -> dict[str, dict]:
    """Collapse per-keyframe description records to one per scene.

    ``SceneDetector.export_metadata`` writes N keyframe rows per scene (N=3 by
    default); the LLM description step independently captions every one of
    them, so ``scene_descriptions.json`` holds N caption records per scene,
    not one. Naively keying a dict comprehension by ``scene_id`` silently
    keeps whichever record happens to be last in the file for that scene --
    an accident of processing/resume order, unrelated to which keyframe is
    actually shown as the scene's representative image.

    This picks the positional-middle record per scene (matching the
    ``kf_meta`` dedup convention used elsewhere for the representative
    keyframe image), then folds any genuinely new sentences from the sibling
    keyframes' captions into its ``description`` text -- each sibling already
    paid the LLM inference cost, so their real content is preserved instead
    of discarded outright. The canonical record's own validity (``error`` key
    / broken-placeholder text) is left untouched by enrichment, so "does this
    scene still need review" keeps tracking the same record that gets shown.

    Returns ``{scene_id_key: record, ...}`` -- same shape/keys the previous
    ``{d["scene_id"]: d for d in descriptions}`` one-liner returned, so no
    downstream consumer needs to change how it reads a record.
    """
    groups: dict[str, list[dict]] = {}
    for entry in descriptions:
        if "scene_id" not in entry:
            continue
        groups.setdefault(scene_id_key(entry["scene_id"]), []).append(entry)

    canonical: dict[str, dict] = {}
    for sid, group in groups.items():
        ordered = [
            rec for _, rec in sorted(enumerate(group), key=lambda pair: _kf_pos(pair[1], pair[0]))
        ]
        mid = len(ordered) // 2
        rep = dict(ordered[mid])
        siblings = ordered[:mid] + ordered[mid + 1 :]

        if "error" not in rep and rep.get("description"):
            normalized = [" ".join(s.lower().split()) for s in _split_sentences(rep["description"])]
            additions: list[str] = []
            for sibling in siblings:
                if "error" in sibling:
                    continue
                for sentence in _split_sentences(sibling.get("description", "")):
                    if _is_novel(sentence, normalized):
                        additions.append(sentence)
                        normalized.append(" ".join(sentence.lower().split()))
            if additions:
                rep["description"] = " ".join([rep["description"].strip(), *additions])

        canonical[sid] = rep

    return canonical


def save_description(ctx: FilmContext, scene_id: int, new_text: str) -> None:
    """Update (or create) the description for ``scene_id`` in ``scene_descriptions.json``.

    Finds the entry whose ``scene_id`` field matches ``scene_id`` and
    replaces its ``description`` value with ``new_text``, preserving all
    other fields (e.g. ``tags``, ``objects``). If no entry exists for
    that scene, a minimal ``{"scene_id": scene_id, "description": new_text}``
    record is appended. The write is atomic (same-dir temp + os.replace)
    with the same permissions semantics as ``kuaa.annotations.io.save``.
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
