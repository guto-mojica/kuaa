"""Scene-cut review + edit service (HTTP-agnostic).

The authoritative representation of scene boundaries is the ``CutSet`` in
``metadata/scene_cuts.json``. Every read builds a filmstrip view model from
``keyframes_metadata.json`` (the render source) plus the cut set (provenance +
stats); every edit mutates the cut set, rebuilds keyframes + metadata from it,
and invalidates the downstream visual/embeddings/LLM artifacts that a corrected
boundary would otherwise leave stale.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kuaa.library import derive_fps, keyframe_url, load_json, to_smpte
from kuaa.scene_detector import (
    CutSet,
    SceneCut,
    SceneDetector,
    read_cutset,
    write_cutset,
)

if TYPE_CHECKING:
    from kuaa.config import Settings
    from kuaa.library import FilmContext

logger = logging.getLogger(__name__)


class CutEditError(ValueError):
    """Raised when a split/merge request is invalid (bad frame, no such cut)."""


# ─── Read path ────────────────────────────────────────────────────────────────


def _grouped_scenes(kf_meta: list) -> list[dict[str, Any]]:
    """Collapse the 1:N keyframe rows to one representative row per scene.

    Mirrors ``kuaa.annotations.scenes.build_scene_list``: the representative
    is the middle keyframe of each scene's group, ordered by ``scene_id``.
    """
    groups: dict[Any, list] = {}
    for entry in kf_meta:
        sid = entry.get("scene_id")
        if sid is not None:
            groups.setdefault(sid, []).append(entry)

    reps: list[dict[str, Any]] = []
    for sid in sorted(groups, key=lambda s: int(s)):
        group = groups[sid]
        rep = dict(group[len(group) // 2])
        reps.append(rep)
    return reps


def load_or_derive_cutset(ctx: FilmContext) -> CutSet | None:
    """Return the persisted cut set, or synthesize one from keyframe metadata.

    Films processed before ``scene_cuts.json`` existed still have
    ``keyframes_metadata.json``; deriving a cut set from it keeps them
    editable. Returns ``None`` only when neither artifact is present.
    """
    cuts_path = ctx.metadata_dir / "scene_cuts.json"
    cutset = read_cutset(cuts_path)
    if cutset is not None:
        return cutset

    raw_kf = load_json(ctx.metadata_dir / "keyframes_metadata.json")
    kf_meta = raw_kf if isinstance(raw_kf, list) else []
    if not kf_meta:
        return None

    reps = _grouped_scenes(kf_meta)
    fps = derive_fps(kf_meta)
    total_frames = max((int(r.get("end_frame") or 0) for r in reps), default=0)
    duration_s = max((float(r.get("end_time_s") or 0.0) for r in reps), default=0.0)
    # Interior cuts are the start of every scene except the first.
    cuts = [
        SceneCut(
            frame=int(r.get("start_frame") or 0),
            time_s=float(r.get("start_time_s") or 0.0),
            source="auto",
        )
        for r in reps[1:]
    ]
    return CutSet(
        fps=fps,
        total_frames=total_frames,
        duration_s=duration_s,
        cuts=cuts,
        params={},
    )


def build_filmstrip(ctx: FilmContext) -> dict[str, Any]:
    """Build the filmstrip view model for the Pre-processing surface.

    Returns a dict with ``scenes`` (one entry per scene, with a representative
    keyframe URL, timecodes, frame range, and the provenance of the cut that
    opens it), aggregate ``stats``, and top-level flags. When no scenes exist
    yet, ``scenes`` is empty and ``has_scenes`` is ``False``.
    """
    raw_kf = load_json(ctx.metadata_dir / "keyframes_metadata.json")
    kf_meta = raw_kf if isinstance(raw_kf, list) else []
    reps = _grouped_scenes(kf_meta)
    cutset = load_or_derive_cutset(ctx)

    # Map interior-cut frame → provenance, to label each scene's opening cut.
    cut_source_by_frame: dict[int, str] = {}
    if cutset is not None:
        cut_source_by_frame = {c.frame: c.source for c in cutset.cuts}

    scenes: list[dict[str, Any]] = []
    durations: list[float] = []
    for idx, rep in enumerate(reps):
        start_s = float(rep.get("start_time_s") or 0.0)
        end_s = float(rep.get("end_time_s") or 0.0)
        duration_s = max(0.0, end_s - start_s)
        durations.append(duration_s)
        start_frame = int(rep.get("start_frame") or 0)
        fps = cutset.fps if cutset else derive_fps(kf_meta)
        scenes.append(
            {
                "scene_id": int(rep.get("scene_id") or (idx + 1)),
                "index": idx + 1,
                "start_frame": start_frame,
                "end_frame": int(rep.get("end_frame") or 0),
                "start_time_s": start_s,
                "end_time_s": end_s,
                "duration_s": duration_s,
                "start_smpte": to_smpte(start_s, fps),
                "keyframe_url": keyframe_url(Path(rep.get("filepath", "")), ctx.data_dir) or "",
                # Provenance of the cut that OPENS this scene ("" for scene 1).
                "cut_source": "" if idx == 0 else cut_source_by_frame.get(start_frame, "auto"),
            }
        )

    stats = _stats(durations)
    return {
        "has_scenes": bool(scenes),
        "scenes": scenes,
        "stats": stats,
        "fps": cutset.fps if cutset else 24.0,
        "total_frames": cutset.total_frames if cutset else 0,
        "duration_s": cutset.duration_s if cutset else 0.0,
        "has_manual_edits": bool(cutset and cutset.has_manual_edits),
        "params": dict(cutset.params) if cutset else {},
    }


def _stats(durations: list[float]) -> dict[str, float | int]:
    if not durations:
        return {"num_scenes": 0}
    ordered = sorted(durations)
    n = len(ordered)
    mid = n // 2
    median = ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    return {
        "num_scenes": n,
        "total_duration_s": sum(durations),
        "mean_s": sum(durations) / n,
        "median_s": median,
        "min_s": ordered[0],
        "max_s": ordered[-1],
    }


# ─── Edit path ────────────────────────────────────────────────────────────────


def _detector_for_rebuild(cfg: Settings, cutset: CutSet) -> SceneDetector:
    """A detector configured to reproduce this film's keyframe extraction.

    Detection thresholds are irrelevant on rebuild (no detection runs); only
    the keyframe-extraction settings matter, and those come from the cut set's
    recorded params so a re-extraction matches the original run.
    """
    det = SceneDetector(cfg)
    params = cutset.params or {}
    det.keyframes_per_scene = int(params.get("keyframes_per_scene", det.keyframes_per_scene))
    det.keyframe_height = int(params.get("keyframe_height", det.keyframe_height))
    return det


def _clear_downstream(ctx: FilmContext, cfg: Settings) -> None:
    """Remove visual/embeddings/LLM artifacts made stale by a boundary change.

    Deliberately does NOT touch keyframes or ``scene_cuts.json`` — the rebuild
    regenerates the keyframes, and the cut set is the source of truth. This is
    the downstream-only sibling of ``api.jobs._clear_scene_detection_cascade``,
    kept here so the core stays free of the HTTP layer.
    """
    for name in ("visual_analysis.json", "scene_descriptions.json", "scene_tags.json"):
        p = ctx.metadata_dir / name
        if p.exists():
            p.unlink()
            logger.info("preprocess: cleared stale %s", p.name)

    emb_cfg = getattr(cfg, "embeddings", None)
    emb_filename = getattr(emb_cfg, "filename", "keyframe_embeddings.npy")
    mapping_filename = getattr(emb_cfg, "mapping_filename", "index_mapping.json")
    for name in (emb_filename, mapping_filename):
        p = ctx.embeddings_dir / name
        if p.exists():
            p.unlink()
            logger.info("preprocess: cleared stale %s", p.name)


def _apply(ctx: FilmContext, cfg: Settings, video_path: Path, cutset: CutSet) -> dict[str, Any]:
    """Persist a mutated cut set, rebuild artifacts, invalidate downstream."""
    metadata_path = ctx.metadata_dir / "keyframes_metadata.json"
    keyframes_dir = ctx.frames_dir / "scenes" / "keyframes_content"
    cuts_path = ctx.metadata_dir / "scene_cuts.json"

    _clear_downstream(ctx, cfg)
    detector = _detector_for_rebuild(cfg, cutset)
    detector.rebuild(cutset, video_path, keyframes_dir, metadata_path)
    write_cutset(cutset, cuts_path)
    return build_filmstrip(ctx)


def split_scene(
    ctx: FilmContext, *, cfg: Settings, video_path: Path, at_frame: int
) -> dict[str, Any]:
    """Split a scene by adding a manual cut at ``at_frame``.

    ``at_frame`` must fall strictly inside an existing scene and not coincide
    with a boundary. Returns the rebuilt filmstrip view model.
    """
    cutset = load_or_derive_cutset(ctx)
    if cutset is None:
        raise CutEditError("No detected scenes to split.")

    at_frame = int(at_frame)
    existing = {c.frame for c in cutset.cuts}
    if at_frame in existing or at_frame <= 0 or at_frame >= cutset.total_frames:
        raise CutEditError(f"Cannot split at frame {at_frame}.")
    # The frame must be interior to some scene (it always is once it is neither
    # a boundary nor the 0/total bookend), so no further containment check is
    # needed — adding the cut simply subdivides whichever scene contains it.

    time_s = at_frame / cutset.fps if cutset.fps else 0.0
    cutset.cuts.append(SceneCut(frame=at_frame, time_s=time_s, source="manual"))
    logger.info("preprocess: split at frame %d", at_frame)
    return _apply(ctx, cfg, video_path, cutset)


def merge_at(
    ctx: FilmContext, *, cfg: Settings, video_path: Path, cut_frame: int
) -> dict[str, Any]:
    """Merge two scenes by removing the boundary cut at ``cut_frame``.

    Returns the rebuilt filmstrip view model.
    """
    cutset = load_or_derive_cutset(ctx)
    if cutset is None:
        raise CutEditError("No detected scenes to merge.")

    cut_frame = int(cut_frame)
    before = len(cutset.cuts)
    cutset.cuts = [c for c in cutset.cuts if c.frame != cut_frame]
    if len(cutset.cuts) == before:
        raise CutEditError(f"No cut at frame {cut_frame} to merge.")
    logger.info("preprocess: merged at frame %d", cut_frame)
    return _apply(ctx, cfg, video_path, cutset)
