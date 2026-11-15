"""Pin a graded pool to the scene numbering it was built against.

``pool_candidates`` promises that "the judgments outlive any one retriever",
and its persistence — ``(film_slug, scene_id, pool)`` — delivers that. What it
does not survive is a change to what ``scene_id`` *means*.

``scene_id`` is an ordinal produced by scene detection: the Nth scene of the
film, renumbered from 1 on every rebuild. Correcting a cut is the headline
workflow of the Pre-processing tab, and merging or splitting one boundary
renumbers every scene after it. A grade recorded before that edit then points
at a different scene, or at none — silently, because nothing on either side of
the join knows the numbering moved. ``preprocess.service._migrate_scene_id_
overrides`` already concedes exactly this for the two other per-scene
artifacts; grades are the third.

This module is the loud half of the fix. A pool file records the scene
manifest hash of every film it drew candidates from; joining grades to that
pool re-computes the hashes and refuses when they differ. The quiet half —
relabelling grades through the ``old_to_new`` map an edit already computes —
lives in ``preprocess.service``, and only covers edits that go through
``apply_pending``. Re-detection does not, which is why the guard exists and is
not merely a nicety on top of the migration.

The durable answer is to content-address scene identity from
``(film, boundary timestamps)``, which would let a cut correction remap grades
by timestamp overlap instead of dropping them. That is a larger change; this
turns silent rot into an error in the meantime.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

#: Key under which a pool file carries ``{film_slug: manifest_hash}``.
MANIFEST_KEY = "scene_manifests"


class SceneManifestMismatch(Exception):
    """A film's scene numbering changed since the pool was generated."""


def scene_manifest_hash(library_dir: Path, slug: str) -> str | None:
    """Hash the ``scene_id → (start_frame, end_frame)`` mapping of one film.

    Computed from ``keyframes_metadata.json`` rather than ``scene_cuts.json``:
    that file is what ``scene_id`` is an ordinal over, it exists for every
    processed film (including ones predating the cut list), and *both* editing
    paths rewrite it — ``apply_pending`` and a plain re-detection. Hashing the
    cut list would miss the second.

    Returns ``None`` when the film has no readable keyframe metadata, which the
    callers treat as "nothing to pin" rather than as a mismatch: a hermetic
    fixture and an unprocessed film both land here, and neither is evidence
    that a numbering changed.
    """
    path = library_dir / slug / "metadata" / "keyframes_metadata.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, list):
        return None

    boundaries: dict[int, tuple[int, int]] = {}
    for entry in raw:
        if not isinstance(entry, dict) or entry.get("scene_id") is None:
            continue
        try:
            sid = int(entry["scene_id"])
            start = int(entry.get("start_frame") or 0)
            end = int(entry.get("end_frame") or 0)
        except (TypeError, ValueError):
            continue
        boundaries[sid] = (start, end)
    if not boundaries:
        return None

    material = ";".join(f"{sid}:{s}-{e}" for sid, (s, e) in sorted(boundaries.items()))
    return hashlib.blake2b(material.encode("utf-8"), digest_size=16).hexdigest()


def build_manifest(library_dir: Path, slugs: list[str]) -> dict[str, str]:
    """``{slug: hash}`` for every slug that has one. Films without are omitted."""
    out: dict[str, str] = {}
    for slug in sorted(set(slugs)):
        digest = scene_manifest_hash(library_dir, slug)
        if digest is not None:
            out[slug] = digest
    return out


def manifest_drift(stored: dict[str, str], library_dir: Path) -> dict[str, tuple[str, str | None]]:
    """Films whose current hash differs from ``stored``.

    Returns ``{slug: (stored_hash, current_hash)}``; ``current_hash`` is
    ``None`` when the film's metadata has become unreadable, which counts as
    drift — a pool cannot be joined to scenes that are no longer there.
    """
    drifted: dict[str, tuple[str, str | None]] = {}
    for slug, stored_hash in sorted(stored.items()):
        current = scene_manifest_hash(library_dir, slug)
        if current != stored_hash:
            drifted[slug] = (stored_hash, current)
    return drifted


def require_manifest_match(stored: dict[str, str], library_dir: Path, *, run: str) -> None:
    """Raise :class:`SceneManifestMismatch` when any film's numbering moved.

    ``stored`` empty (a pool written before this guard existed, or a hermetic
    fixture) is a pass with a warning: the guard cannot retroactively pin a
    file that never recorded a hash, and refusing to grade every such pool
    would be a worse outcome than saying so.
    """
    if not stored:
        logger.warning(
            "eval run %r carries no scene manifest — its grades are not pinned to a "
            "scene numbering; regenerate the pool to pin them",
            run,
        )
        return
    drifted = manifest_drift(stored, library_dir)
    if not drifted:
        return
    detail = ", ".join(
        f"{slug} (pool {old[:8]} → now {(new or 'missing')[:8]})"
        for slug, (old, new) in drifted.items()
    )
    raise SceneManifestMismatch(
        f"scene numbering changed since eval run {run!r} was generated: {detail}. "
        f"Every grade for these films is keyed to the old ordinals. Re-run "
        f"'kuaa eval slate' to rebuild the pool, or restore the previous cuts."
    )


__all__ = [
    "MANIFEST_KEY",
    "SceneManifestMismatch",
    "build_manifest",
    "manifest_drift",
    "require_manifest_match",
    "scene_manifest_hash",
]
