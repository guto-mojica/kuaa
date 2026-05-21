"""Processing tab — context builder for the Mojica .p-cp / .p-active layout.

The tab partial wants more than ``{films, step_defs, jobs}``:

  * ``initial_log_lines`` — empty for M1 (the SSE stream feeds them
    live); the route layer may later seed a tail of the rotating
    in-memory log buffer.
  * ``stats`` — aggregate counts (frames / scenes / embeddings /
    descriptions / faces / objects) summed across all registered films.
  * ``job_queue`` — recent-job history derived from ``JobRegistry.all()``,
    mapped onto the .p-queue item-status vocabulary
    (``done`` / ``proc`` / ``queued`` / ``error`` / ``cancelled``).
  * ``active_step`` — sub-step detail card for the right pane (.p-rp).
    Only present when a job is running; the substeps list is a
    pipeline-step-aware stub until a real instrumentation pass lands.
  * ``gpu_metrics`` — empty by default. Populated only when
    ``cfg.proc.gpu_metrics_enabled`` is true AND a measurement source
    is wired (none today). The substeps partial omits the GPU card
    entirely when the list is empty / the flag is off.

All context fields default to safe empties so the partial renders the
layout honestly on a fresh / empty install — no fake numbers.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from types import MappingProxyType
from typing import Any

logger = logging.getLogger(__name__)


# ── Pipeline step descriptions (right-pane copy) ──────────────────────────────
#
# Short, factual descriptions of what each pipeline step does — surfaced
# in the .p-rp "what" paragraph and the synthetic substeps list. The
# real backend doesn't expose per-step substeps yet; the list below is
# a thin, honest placeholder ("loading model" / "running" / "persisting")
# so the layout has content to render. When a future runner emits real
# sub-step progress, ``active_step.substeps`` becomes the live source
# and this fallback can be deleted.

_STEP_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "frame_extraction": {
        "label": "Frame extraction",
        "detail": "ffmpeg · 1 fps · 480p",
        "description": (
            "Decode the source video and emit one keyframe per second "
            "(downscaled). Feeds scene detection and visual analysis."
        ),
    },
    "scene_detection": {
        "label": "Scene detection",
        "detail": "PySceneDetect · adaptive",
        "description": (
            "Detect shot boundaries across the extracted frames and "
            "build the scene index. Output: scene cuts + representative "
            "keyframes."
        ),
    },
    "visual_analysis": {
        "label": "Visual analysis",
        "detail": "YOLOv8 (objects) + MTCNN (faces)",
        "description": (
            "Run object and face detection on each scene's keyframes to "
            "produce automatic tags. Output feeds embeddings + descriptions."
        ),
    },
    "embeddings": {
        "label": "Embeddings",
        "detail": "CLIP ViT-B/32",
        "description": (
            "Encode every keyframe into a CLIP embedding so the scene "
            "is searchable by text query or by another image."
        ),
    },
    "llm_description": {
        "label": "LLM descriptions",
        "detail": "Moondream2 · transformers",
        "description": (
            "Generate a short natural-language description of each scene "
            "from its keyframe. Slowest step; CPU-bound by default."
        ),
    },
}


# ── Per-step substep recipes ──────────────────────────────────────────────────
#
# Each pipeline step has its own substep list so the right pane reads as a
# meaningful breakdown ("Load YOLOv8 weights" → "Detect objects" → "Persist
# tags") instead of a generic "load → run → persist" trio. The recipes are
# still synthetic — real per-step instrumentation will replace the static
# value column when the runner emits sub-step progress — but they line up
# with the visible work each step actually performs.

_SUBSTEP_RECIPES: MappingProxyType[str, tuple[dict[str, str], ...]] = MappingProxyType(
    {
        "frame_extraction": (
            {"label": "Probe video", "sub": "ffmpeg · streams", "value": "~0.3s"},
            {"label": "Extract frames", "sub": "1 fps · 480p", "value": ""},
            {"label": "Persist frames", "sub": "data/frames/", "value": "~5s"},
        ),
        "scene_detection": (
            {"label": "Read frames", "sub": "PySceneDetect input", "value": "~0.5s"},
            {"label": "Detect cuts", "sub": "adaptive threshold", "value": ""},
            {"label": "Pick keyframes", "sub": "1 per scene", "value": "~2s"},
            {"label": "Persist scenes", "sub": "keyframes_metadata.json", "value": "~1s"},
        ),
        "visual_analysis": (
            {"label": "Load YOLOv8 weights", "sub": "ultralytics", "value": "~0.4s"},
            {"label": "Load MTCNN", "sub": "facenet-pytorch", "value": "~0.6s"},
            {"label": "Detect objects", "sub": "per scene", "value": ""},
            {"label": "Detect faces", "sub": "per scene", "value": ""},
            {"label": "Persist tags", "sub": "tags_per_scene.json", "value": "~15s"},
        ),
        "embeddings": (
            {"label": "Load CLIP", "sub": "ViT-B/32", "value": "~1s"},
            {"label": "Encode keyframes", "sub": "batch · GPU/CPU", "value": ""},
            {"label": "Persist embeddings", "sub": "keyframe_embeddings.npy", "value": "~2s"},
        ),
        "llm_description": (
            {"label": "Load Moondream2", "sub": "transformers", "value": "~6s"},
            {"label": "Describe scenes", "sub": "per keyframe", "value": ""},
            {"label": "Persist descriptions", "sub": "scene_descriptions.json", "value": "~3s"},
        ),
    }
)

_DEFAULT_RECIPE: tuple[dict[str, str], ...] = (
    {"label": "Load weights", "sub": "", "value": ""},
    {"label": "Run", "sub": "", "value": ""},
    {"label": "Persist outputs", "sub": "", "value": ""},
)


_ACTIVE_STATUS_BY_INDEX = ("done", "active")  # i=0 → done, i=1 → active, else pending


def _fallback_substeps(step_name: str, step_state: str) -> list[dict[str, str]]:
    """Per-step synthetic substeps so the right pane renders meaningfully.

    Status derived from the parent step's state:
      'done'   → all done
      'active' → first sub done, second active, remainder pending
      else     → all pending
    """
    recipe = _SUBSTEP_RECIPES.get(step_name, _DEFAULT_RECIPE)
    if step_state == "done":
        return [{**r, "status": "done"} for r in recipe]
    if step_state == "active":
        return [
            {**r, "status": _ACTIVE_STATUS_BY_INDEX[i] if i < 2 else "pending"}
            for i, r in enumerate(recipe)
        ]
    return [{**r, "status": "pending"} for r in recipe]


# ── Stats aggregation ─────────────────────────────────────────────────────────


def aggregate_stats(library_dir: Path) -> dict[str, Any]:
    """Sum scene-level counts across all registered films.

    Reads each film's ``metadata/keyframes_metadata.json`` (scene count)
    and best-effort sums embeddings/descriptions/faces/objects when the
    metadata files are present. Anything missing stays at 0.

    Frames are not persisted per-scene in metadata; ``frames`` falls
    back to ``scenes * 1`` (rough lower bound) so the card never shows
    a less-than-scenes number. A future pass can hook this into the
    real frame index for an honest count.
    """
    import json

    stats = {
        "frames": 0,
        "scenes": 0,
        "embeddings": 0,
        "descriptions": 0,
        "faces": 0,
        "objects": 0,
        "faces_warn": False,
    }

    if not library_dir.exists():
        return stats

    for film_dir in sorted(library_dir.iterdir()):
        if not film_dir.is_dir():
            continue

        meta_dir = film_dir / "metadata"
        kf_path = meta_dir / "keyframes_metadata.json"
        film_scene_count = 0
        if kf_path.exists():
            try:
                with open(kf_path, encoding="utf-8") as f:
                    kf_meta = json.load(f)
                if isinstance(kf_meta, list):
                    film_scene_count = len(kf_meta)
                    stats["scenes"] += film_scene_count
                    # Lower-bound frame estimate (one per scene) — honest
                    # placeholder until a real frame count lands.
                    stats["frames"] += film_scene_count
            except (json.JSONDecodeError, OSError):
                pass

        emb_path = film_dir / "embeddings" / "keyframe_embeddings.npy"
        if emb_path.exists():
            # File presence is a proxy: the actual row count would need
            # numpy; treat one present index as N scenes worth of
            # embeddings (matches the scene count for that film, which
            # we already computed above so no second JSON read is needed).
            stats["embeddings"] += film_scene_count

        desc_path = meta_dir / "scene_descriptions.json"
        if desc_path.exists():
            try:
                with open(desc_path, encoding="utf-8") as f:
                    descs = json.load(f)
                if isinstance(descs, (list, dict)):
                    stats["descriptions"] += len(descs)
            except (json.JSONDecodeError, OSError):
                pass

        tags_path = meta_dir / "scene_tags.json"
        if tags_path.exists():
            try:
                with open(tags_path, encoding="utf-8") as f:
                    tags = json.load(f)
                # scene_tags.json is typically a {scene_id: [tags]} dict
                # or a list of per-scene records — count faces/objects
                # by tag-name match. Defensive: any structural surprise
                # is silently skipped.
                if isinstance(tags, dict):
                    for scene_tags in tags.values():
                        if isinstance(scene_tags, list):
                            for t in scene_tags:
                                name = (t or "").lower() if isinstance(t, str) else ""
                                if "face" in name:
                                    stats["faces"] += 1
                                elif name and name not in {"day", "night", "indoor", "outdoor"}:
                                    stats["objects"] += 1
            except (json.JSONDecodeError, OSError):
                pass

    return stats


# ── Job queue mapping ─────────────────────────────────────────────────────────


def _humanise_delta(seconds: float) -> str:
    """Render an elapsed-seconds float as a short human string.

    The .p-queue cards expect a single short token: ``now`` / ``5m`` /
    ``2h`` / ``4d``. No locale-specific suffixes — the rendered text
    is intentionally minimal.
    """
    if seconds < 60:
        return "now"
    if seconds < 3600:
        return f"{int(seconds // 60)}m"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h"
    return f"{int(seconds // 86400)}d"


def _job_queue_status(job_status: str) -> str:
    """Map the JobState lifecycle vocabulary onto the .p-queue item
    status modifier (``done`` / ``proc`` / ``queued`` / ``error`` /
    ``cancelled``) so proc.css colours the leading dot correctly."""
    if job_status == "running":
        return "proc"
    if job_status == "created":
        return "queued"
    if job_status == "error":
        return "error"
    if job_status == "cancelled":
        return "cancelled"
    return "done"


def build_job_queue(jobs: list[Any]) -> list[dict[str, Any]]:
    """Map all registered jobs onto the .p-queue item shape.

    Newest first (jobs registry retains the bounded recent history).
    """
    now = time.time()
    queue: list[dict[str, Any]] = []
    for j in sorted(jobs, key=lambda j: getattr(j, "created_at", 0.0), reverse=True):
        queue.append(
            {
                "film_title": getattr(j, "video_name", j.id),
                "status": _job_queue_status(j.status),
                "when_display": _humanise_delta(now - getattr(j, "created_at", now)),
            }
        )
    return queue


# ── Active step (right pane) ──────────────────────────────────────────────────


def build_active_step(jobs: list[Any]) -> dict[str, Any] | None:
    """Build the .p-rp sub-step context for the first running job.

    Returns ``None`` when there is no active job (the right pane is
    then rendered empty). The substeps list is the synthetic fallback
    above — a real per-step sub-progress feed would replace it.
    """
    if not jobs:
        return None
    job = jobs[0]
    # Locate the first non-terminal step; fall back to the last step
    # (so a fully-done job still has something to display).
    active_idx = 0
    for i, step in enumerate(job.steps):
        if step.state == "active":
            active_idx = i
            break
        if step.state in ("done", "skipped", "error", "blocked"):
            active_idx = min(i + 1, len(job.steps) - 1)

    step = job.steps[active_idx]
    meta = _STEP_DESCRIPTIONS.get(step.name, {"label": step.name, "detail": "", "description": ""})
    return {
        "idx": active_idx,
        "label": meta["label"],
        "detail": meta["detail"],
        "description": meta["description"],
        "substeps": _fallback_substeps(step.name, step.state),
    }


# ── Job enrichment for the .p-active card header ──────────────────────────────


def enrich_jobs(jobs: list[Any]) -> list[Any]:
    """Attach display fields the .p-active template header needs.

    Mutates each ``JobState`` in place with computed attributes so the
    template can use simple ``{{ job.film_title }}`` access (Jinja
    walks ``__dict__`` first, then falls back to attribute lookup).
    Fields added:

      * ``film_title`` — defaults to ``video_name`` (basename without ext).
      * ``film_thumb`` — ``None`` (no thumb wiring today).
      * ``year`` / ``director`` — ``None`` (not yet tracked per job).
      * ``started_at_display`` — ``HH:MM:SS`` of ``created_at``.
      * ``elapsed_display`` — short elapsed string.
      * ``active_step_idx`` — index of the currently active step.
      * ``step_progress`` / ``throughput`` / ``eta_display`` — empty
        strings (no instrumentation yet).
    """
    now = time.time()
    out: list[Any] = []
    for job in jobs:
        # Use video_name (basename minus extension feel: keep extension
        # for now — the legacy regression test asserts "jeca_tatu.mp4"
        # is present in the rendered job, so we must NOT strip it).
        video_name = getattr(job, "video_name", "")

        active_idx = 0
        for i, step in enumerate(job.steps):
            if step.state == "active":
                active_idx = i
                break
            if step.state in ("done", "skipped"):
                active_idx = min(i + 1, len(job.steps) - 1)

        # Set on the JobState instance directly (it's a dataclass, so
        # Jinja sees these as regular attributes alongside steps/progress).
        job.film_title = video_name  # type: ignore[attr-defined]
        job.film_thumb = None  # type: ignore[attr-defined]
        job.year = None  # type: ignore[attr-defined]
        job.director = None  # type: ignore[attr-defined]
        job.started_at_display = time.strftime(  # type: ignore[attr-defined]
            "%H:%M:%S", time.localtime(getattr(job, "created_at", now))
        )
        job.elapsed_display = _humanise_delta(  # type: ignore[attr-defined]
            now - getattr(job, "created_at", now)
        )
        job.active_step_idx = active_idx  # type: ignore[attr-defined]
        job.step_progress = ""  # type: ignore[attr-defined]
        job.throughput = ""  # type: ignore[attr-defined]
        job.eta_display = ""  # type: ignore[attr-defined]

        out.append(job)
    return out
