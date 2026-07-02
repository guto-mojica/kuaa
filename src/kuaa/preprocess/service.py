"""Scene-cut review + edit service (HTTP-agnostic).

The authoritative representation of scene boundaries is the ``CutSet`` in
``metadata/scene_cuts.json``. Every read builds a filmstrip view model from
``keyframes_metadata.json`` (the render source) plus the cut set (provenance +
stats); every edit mutates the cut set, rebuilds keyframes + metadata from it,
and invalidates the downstream visual/embeddings/LLM artifacts that a corrected
boundary would otherwise leave stale.

Manual split/merge edits go through a staging workflow rather than rebuilding
immediately: ``stage_split``/``stage_merge`` append a :class:`PendingOp` to
``pending_edits.json`` (a cheap JSON write, no rebuild), the operator reviews
the staged list (each op individually checkable), and ``apply_pending`` runs
the actual — now partial, not whole-film — rebuild once for whatever is
checked. ``discard_pending`` clears the staged list with no rebuild at all.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

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

PENDING_FILENAME = "pending_edits.json"


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
    """Return the persisted, last-*applied* cut set, or synthesize one.

    Films processed before ``scene_cuts.json`` existed still have
    ``keyframes_metadata.json``; deriving a cut set from it keeps them
    editable. Returns ``None`` only when neither artifact is present. This is
    deliberately blind to any staged ``pending_edits.json`` — it always
    reflects what's actually rebuilt on disk right now.
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

    Tiles/thumbnails always render from the last-*applied* state (what's
    really rebuilt on disk) — a staged, not-yet-applied edit never changes
    what a scene tile looks like. Pending edits are instead surfaced as
    overlays computed from ``pending_edits.json``: ``pending_removal`` on a
    scene flags that the cut opening it is staged for a (checked) merge;
    ``pending_splits`` lists the relative (0..1) positions inside a scene of
    any staged (checked) split that would divide it. ``stats`` reflect the
    *effective* cut set (applied + checked pending ops replayed), so scene
    count/durations read correctly live even before a rebuild happens.

    Returns a dict with ``scenes``, aggregate ``stats``, top-level flags, and
    the staged-edit list (``pending_ops``, ``pending_count``). When no scenes
    exist yet, ``scenes`` is empty and ``has_scenes`` is ``False``.
    """
    raw_kf = load_json(ctx.metadata_dir / "keyframes_metadata.json")
    kf_meta = raw_kf if isinstance(raw_kf, list) else []
    reps = _grouped_scenes(kf_meta)
    applied = load_or_derive_cutset(ctx)

    cut_source_by_frame: dict[int, str] = {}
    if applied is not None:
        cut_source_by_frame = {c.frame: c.source for c in applied.cuts}

    pending = read_pending_edits(ctx) if applied is not None else PendingEdits()
    checked_ops = _checked_ops(pending)
    effective = replay(applied, checked_ops) if applied is not None else None
    pending_merge_frames = {op.frame for op in checked_ops if op.type == "merge"}
    pending_split_ops = [op for op in checked_ops if op.type == "split"]

    scenes: list[dict[str, Any]] = []
    durations: list[float] = []
    for idx, rep in enumerate(reps):
        start_s = float(rep.get("start_time_s") or 0.0)
        end_s = float(rep.get("end_time_s") or 0.0)
        duration_s = max(0.0, end_s - start_s)
        durations.append(duration_s)
        start_frame = int(rep.get("start_frame") or 0)
        end_frame = int(rep.get("end_frame") or 0)
        fps = applied.fps if applied else derive_fps(kf_meta)
        pending_splits = [
            (op.frame - start_frame) / (end_frame - start_frame)
            for op in pending_split_ops
            if end_frame > start_frame and start_frame < op.frame < end_frame
        ]
        scenes.append(
            {
                "scene_id": int(rep.get("scene_id") or (idx + 1)),
                "index": idx + 1,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time_s": start_s,
                "end_time_s": end_s,
                "duration_s": duration_s,
                "start_smpte": to_smpte(start_s, fps),
                "keyframe_url": keyframe_url(Path(rep.get("filepath", "")), ctx.data_dir) or "",
                # Provenance of the cut that OPENS this scene ("" for scene 1).
                "cut_source": "" if idx == 0 else cut_source_by_frame.get(start_frame, "auto"),
                "pending_removal": idx != 0 and start_frame in pending_merge_frames,
                "pending_splits": pending_splits,
            }
        )

    if effective is not None:
        eff_fps = effective.fps or 24.0
        eff_durations = [(end - start) / eff_fps for start, end in effective.scene_boundaries()]
        stats = _stats(eff_durations)
    else:
        stats = _stats(durations)

    fps_for_labels = applied.fps if applied else 24.0
    pending_ops = [
        {
            "index": i,
            "type": op.type,
            "checked": op.checked,
            "frame": op.frame,
            "start_smpte": to_smpte(op.time_s, fps_for_labels),
        }
        for i, op in enumerate(pending.ops)
    ]

    return {
        "has_scenes": bool(scenes),
        "scenes": scenes,
        "stats": stats,
        "fps": applied.fps if applied else 24.0,
        "total_frames": applied.total_frames if applied else 0,
        "duration_s": applied.duration_s if applied else 0.0,
        "has_manual_edits": bool(applied and applied.has_manual_edits),
        "params": dict(applied.params) if applied else {},
        "pending_ops": pending_ops,
        "pending_count": len(pending.ops),
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


# ─── Staged edits ──────────────────────────────────────────────────────────────


@dataclass
class PendingOp:
    """One staged, not-yet-applied split or merge.

    ``checked`` controls whether :func:`apply_pending` will include it —
    unchecking is how an operator discards a single staged edit without
    clearing the whole session.
    """

    type: Literal["split", "merge"]
    frame: int
    time_s: float
    checked: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "frame": self.frame,
            "time_s": self.time_s,
            "checked": self.checked,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PendingOp:
        return cls(
            type=raw["type"],
            frame=int(raw["frame"]),
            time_s=float(raw.get("time_s", 0.0)),
            checked=bool(raw.get("checked", True)),
        )


@dataclass
class PendingEdits:
    """The staged-edit session for one film. Its presence (non-empty ``ops``)
    is what "pending" means — no separate flag is persisted."""

    ops: list[PendingOp] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"ops": [op.to_dict() for op in self.ops]}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PendingEdits:
        return cls(ops=[PendingOp.from_dict(o) for o in raw.get("ops", [])])


def _pending_path(ctx: FilmContext) -> Path:
    return ctx.metadata_dir / PENDING_FILENAME


def read_pending_edits(ctx: FilmContext) -> PendingEdits:
    """Return the staged-edit session for ``ctx``, or an empty one."""
    path = _pending_path(ctx)
    if not path.exists():
        return PendingEdits()
    with open(path, encoding="utf-8") as f:
        return PendingEdits.from_dict(json.load(f))


def write_pending_edits(pending: PendingEdits, ctx: FilmContext) -> Path:
    """Persist the staged-edit session, or remove the file once it's empty."""
    path = _pending_path(ctx)
    if not pending.ops:
        path.unlink(missing_ok=True)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pending.to_dict(), f, indent=2, ensure_ascii=False)
    return path


def _checked_ops(pending: PendingEdits) -> list[PendingOp]:
    return [op for op in pending.ops if op.checked]


def _apply_op(cutset: CutSet, op: PendingOp) -> CutSet:
    """Mutate ``cutset`` in place by applying ``op``. Raises ``CutEditError``
    if it no longer applies cleanly (bad split frame, or no such cut to
    merge). Callers that want best-effort semantics (see :func:`replay`)
    should catch and skip rather than propagate."""
    if op.type == "split":
        at_frame = op.frame
        existing = {c.frame for c in cutset.cuts}
        if at_frame in existing or at_frame <= 0 or at_frame >= cutset.total_frames:
            raise CutEditError(f"Cannot split at frame {at_frame}.")
        cutset.cuts.append(SceneCut(frame=at_frame, time_s=op.time_s, source="manual"))
    else:
        cut_frame = op.frame
        before = len(cutset.cuts)
        cutset.cuts = [c for c in cutset.cuts if c.frame != cut_frame]
        if len(cutset.cuts) == before:
            raise CutEditError(f"No cut at frame {cut_frame} to merge.")
    return cutset


def replay(base: CutSet, ops: list[PendingOp]) -> CutSet:
    """Apply ``ops`` in order to a copy of ``base``, skipping any that no
    longer apply cleanly.

    Best-effort by design: this powers both the live filmstrip preview and
    ``apply_pending``, so toggling any combination of staged edits on/off
    (e.g. unchecking a split whose frame a later staged merge targets) is
    always safe rather than needing its own validation pass.
    """
    cutset = CutSet(
        fps=base.fps,
        total_frames=base.total_frames,
        duration_s=base.duration_s,
        cuts=[SceneCut(c.frame, c.time_s, c.source) for c in base.cuts],
        params=dict(base.params),
    )
    for op in ops:
        try:
            _apply_op(cutset, op)
        except CutEditError:
            continue
    return cutset


def has_pending(ctx: FilmContext) -> bool:
    return bool(read_pending_edits(ctx).ops)


def stage_split(ctx: FilmContext, *, at_frame: int) -> dict[str, Any]:
    """Stage a split at ``at_frame`` — a cheap JSON append, no rebuild."""
    base = load_or_derive_cutset(ctx)
    if base is None:
        raise CutEditError("No detected scenes to split.")

    pending = read_pending_edits(ctx)
    effective = replay(base, _checked_ops(pending))
    at_frame = int(at_frame)
    time_s = at_frame / effective.fps if effective.fps else 0.0
    op = PendingOp(type="split", frame=at_frame, time_s=time_s)
    _apply_op(effective, op)  # validates against the current staged state

    pending.ops.append(op)
    write_pending_edits(pending, ctx)
    logger.info("preprocess: staged split at frame %d", at_frame)
    return build_filmstrip(ctx)


def stage_merge(ctx: FilmContext, *, cut_frame: int) -> dict[str, Any]:
    """Stage a merge removing the cut at ``cut_frame`` — cheap, no rebuild."""
    base = load_or_derive_cutset(ctx)
    if base is None:
        raise CutEditError("No detected scenes to merge.")

    pending = read_pending_edits(ctx)
    effective = replay(base, _checked_ops(pending))
    cut_frame = int(cut_frame)
    matching = next((c for c in effective.cuts if c.frame == cut_frame), None)
    time_s = matching.time_s if matching else (cut_frame / effective.fps if effective.fps else 0.0)
    op = PendingOp(type="merge", frame=cut_frame, time_s=time_s)
    _apply_op(effective, op)  # validates a cut exists at cut_frame

    pending.ops.append(op)
    write_pending_edits(pending, ctx)
    logger.info("preprocess: staged merge at frame %d", cut_frame)
    return build_filmstrip(ctx)


def toggle_pending(ctx: FilmContext, *, index: int) -> dict[str, Any]:
    """Flip whether the staged op at ``index`` will be applied."""
    pending = read_pending_edits(ctx)
    if not (0 <= index < len(pending.ops)):
        raise CutEditError(f"No pending change at index {index}.")
    pending.ops[index].checked = not pending.ops[index].checked
    write_pending_edits(pending, ctx)
    return build_filmstrip(ctx)


def discard_pending(ctx: FilmContext) -> dict[str, Any]:
    """Clear the staged-edit session with no rebuild. No-op if nothing staged."""
    _pending_path(ctx).unlink(missing_ok=True)
    logger.info("preprocess: discarded pending edits")
    return build_filmstrip(ctx)


# ─── Apply (partial rebuild) ────────────────────────────────────────────────────


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

    Deliberately does NOT touch keyframes or ``scene_cuts.json`` — the apply
    step above already rebuilds those directly. This is the downstream-only
    sibling of ``api.jobs._clear_scene_detection_cascade``, kept here so the
    core stays free of the HTTP layer.
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


def _migrate_scene_id_overrides(ctx: FilmContext, old_to_new: dict[int, int]) -> None:
    """Relabel ``tag_overrides.json`` / ``manual_annotations.json`` keys.

    Both files are keyed by ``scene_id`` but were never migrated or cleared
    on renumbering — any split/merge has always silently left them pointing
    at the wrong scene. This has the old→new mapping already computed for
    the rename step, so it fixes that as a side effect: entries for scenes
    that only shifted position are relabelled; entries for scenes touched or
    orphaned by the edit (absent from the mapping) are dropped, exactly like
    every other per-scene artifact :func:`_clear_downstream` already wipes.
    """
    from kuaa.annotations import io as annotations_io
    from kuaa.annotations import overrides as annotations_overrides

    def _migrate(data: dict[str, Any]) -> dict[str, Any]:
        migrated: dict[str, Any] = {}
        for old_sid_str, value in data.items():
            try:
                old_sid = int(old_sid_str)
            except (TypeError, ValueError):
                continue
            new_sid = old_to_new.get(old_sid)
            if new_sid is not None:
                migrated[str(new_sid)] = value
        return migrated

    annotations = annotations_io.load_annotations(ctx)
    if annotations:
        annotations_io.save_annotations(ctx, _migrate(annotations))

    overrides = annotations_overrides.load_overrides(ctx)
    if overrides:
        annotations_overrides.save_overrides(ctx, _migrate(overrides))


def apply_pending(ctx: FilmContext, *, cfg: Settings, video_path: Path) -> dict[str, Any]:
    """Apply the checked staged edits: partial rebuild + persist + invalidate.

    Only scenes actually touched by an edit (created by a split, or the
    result of a merge) are re-extracted from video; every scene whose
    boundary is unchanged — just shifted to a new position by the edit — is
    renamed on disk, never re-decoded. See module docstring + the plan this
    implements for the full rationale.
    """
    pending = read_pending_edits(ctx)
    checked_ops = _checked_ops(pending)
    if not checked_ops:
        raise CutEditError("No pending changes selected to apply.")

    applied = load_or_derive_cutset(ctx)
    if applied is None:
        raise CutEditError("No detected scenes to edit.")

    final_cutset = replay(applied, checked_ops)
    fps = final_cutset.fps or 24.0

    old_boundaries = applied.scene_boundaries()
    new_boundaries = final_cutset.scene_boundaries()
    old_set, new_set = set(old_boundaries), set(new_boundaries)
    unchanged = old_set & new_set
    orphaned = old_set - new_set

    raw_kf = load_json(ctx.metadata_dir / "keyframes_metadata.json")
    kf_meta = raw_kf if isinstance(raw_kf, list) else []
    reps = _grouped_scenes(kf_meta)
    by_scene_id: dict[int, list[dict]] = {}
    for entry in kf_meta:
        sid = entry.get("scene_id")
        if sid is not None:
            by_scene_id.setdefault(int(sid), []).append(entry)

    old_scene_id_by_boundary: dict[tuple[int, int], int] = {}
    old_files_by_boundary: dict[tuple[int, int], list[Path]] = {}
    for rep in reps:
        boundary = (int(rep.get("start_frame") or 0), int(rep.get("end_frame") or 0))
        sid = int(rep.get("scene_id") or 0)
        old_scene_id_by_boundary[boundary] = sid
        sid_rows = sorted(by_scene_id.get(sid, []), key=lambda r: r.get("keyframe_id", ""))
        old_files_by_boundary[boundary] = [
            Path(r["filepath"]) for r in sid_rows if r.get("filepath")
        ]

    keyframes_dir = ctx.frames_dir / "scenes" / "keyframes_content"
    metadata_path = ctx.metadata_dir / "keyframes_metadata.json"
    cuts_path = ctx.metadata_dir / "scene_cuts.json"
    temp_dir = keyframes_dir.parent / f"{keyframes_dir.name}.new"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    filenames_by_scene_id: dict[int, list[str]] = {}
    old_to_new_scene_id: dict[int, int] = {}
    touched_boundaries: list[tuple[int, int]] = []
    touched_scene_ids: list[int] = []

    for new_idx, boundary in enumerate(new_boundaries):
        new_scene_id = new_idx + 1
        if boundary in unchanged:
            old_scene_id = old_scene_id_by_boundary[boundary]
            old_to_new_scene_id[old_scene_id] = new_scene_id
            names = []
            for pos, src in enumerate(old_files_by_boundary.get(boundary, []), start=1):
                dest = temp_dir / f"scene_{new_scene_id:04d}_kf_{pos:02d}.jpg"
                src.rename(dest)
                names.append(dest.name)
            filenames_by_scene_id[new_scene_id] = names
        else:
            touched_boundaries.append(boundary)
            touched_scene_ids.append(new_scene_id)

    detector = _detector_for_rebuild(cfg, final_cutset)
    extracted = detector.extract_keyframes_for_boundaries(
        touched_boundaries, touched_scene_ids, fps, video_path, temp_dir
    )
    for scene_id, paths in extracted.items():
        filenames_by_scene_id[scene_id] = [p.name for p in paths]

    if keyframes_dir.exists():
        shutil.rmtree(keyframes_dir)
    temp_dir.rename(keyframes_dir)

    logger.info(
        "preprocess: apply — %d unchanged (renamed), %d touched (re-extracted), "
        "%d orphaned; scene_id map: %s",
        len(unchanged),
        len(touched_boundaries),
        len(orphaned),
        old_to_new_scene_id,
    )

    rows: list[dict[str, Any]] = []
    for new_idx, boundary in enumerate(new_boundaries):
        new_scene_id = new_idx + 1
        start, end = boundary
        for pos, filename in enumerate(filenames_by_scene_id.get(new_scene_id, []), start=1):
            rows.append(
                {
                    "scene_id": new_scene_id,
                    "keyframe_id": f"scene_{new_scene_id:04d}_kf_{pos:02d}",
                    "filepath": str(keyframes_dir / filename),
                    "start_time_s": start / fps,
                    "end_time_s": end / fps,
                    "duration_s": (end - start) / fps,
                    "start_frame": start,
                    "end_frame": end,
                }
            )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    _migrate_scene_id_overrides(ctx, old_to_new_scene_id)

    write_cutset(final_cutset, cuts_path)
    _pending_path(ctx).unlink(missing_ok=True)
    _clear_downstream(ctx, cfg)
    return build_filmstrip(ctx)
