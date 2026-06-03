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
  * ``gpu_metrics`` — resource metrics for the right pane. Populated when
    ``cfg.proc.gpu_metrics_enabled`` is true from local psutil CPU/RAM data
    and, when available, already-loaded torch CUDA memory data. The substeps
    partial omits the card when the list is empty / the flag is off.

All context fields default to safe empties so the partial renders the
layout honestly on a fresh / empty install — no fake numbers.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from api.services.processing_stats import (
    _fallback_substeps,
    aggregate_stats,
    build_resource_metrics,
)

logger = logging.getLogger(__name__)

# Re-export the moved symbols so existing import sites keep working.
__all__ = [
    "aggregate_stats",
    "build_resource_metrics",
    "_fallback_substeps",
    "build_active_step",
    "build_job_queue",
    "_job_queue_status",
    "_humanise_delta",
    "enrich_jobs",
]


# ── Pipeline step descriptions (right-pane copy) ──────────────────────────────
#
# Short, factual descriptions of what each pipeline step does — surfaced
# in the .p-rp "what" paragraph and the synthetic substeps list. The
# real backend doesn't expose per-step substeps yet; the list below is
# a thin, honest fallback ("loading model" / "running" / "persisting")
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


# ── Job queue mapping ─────────────────────────────────────────────────────────


def _humanise_delta(seconds: float) -> str:  # "now" / "5m" / "2h" / "4d"
    if seconds < 60:
        return "now"
    if seconds < 3600:
        return f"{int(seconds // 60)}m"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h"
    return f"{int(seconds // 86400)}d"


def _job_queue_status(job_status: str) -> str:  # maps lifecycle vocab → proc.css dot colour
    if job_status == "running":
        return "proc"
    if job_status == "created":
        return "queued"
    if job_status == "error":
        return "error"
    if job_status == "cancelled":
        return "cancelled"
    return "done"


def build_job_queue(  # pending first, then active/terminal newest-first
    jobs: list[Any], *, pending: list[Any] | None = None
) -> list[dict[str, Any]]:
    now = time.time()
    result: list[dict[str, Any]] = []

    for entry in pending or []:
        result.append(
            {
                "film_title": getattr(entry, "video_name", entry.id),
                "status": "queued",
                "when_display": _humanise_delta(now - getattr(entry, "created_at", now)),
                "entry_id": entry.id,
                "is_pending": True,
            }
        )

    for j in sorted(jobs, key=lambda j: getattr(j, "created_at", 0.0), reverse=True):
        result.append(
            {
                "film_title": getattr(j, "video_name", j.id),
                "status": _job_queue_status(j.status),
                "when_display": _humanise_delta(now - getattr(j, "created_at", now)),
                "entry_id": None,
                "is_pending": False,
            }
        )
    return result


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
        job.film_title = video_name
        job.film_thumb = None
        job.year = None
        job.director = None
        job.started_at_display = time.strftime(
            "%H:%M:%S", time.localtime(getattr(job, "created_at", now))
        )
        job.elapsed_display = _humanise_delta(now - getattr(job, "created_at", now))
        job.active_step_idx = active_idx
        job.step_progress = ""
        job.throughput = ""
        job.eta_display = ""

        out.append(job)
    return out
