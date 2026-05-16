"""In-memory job registry and pipeline runner for the Processing tab."""
from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Optional

logger = logging.getLogger(__name__)

STEP_DEFS: list[tuple[str, str]] = [
    ("frame_extraction", "Frames"),
    ("scene_detection",  "Cenas"),
    ("visual_analysis",  "Visual"),
    ("embeddings",       "Embeddings"),
    ("llm_description",  "Descrições"),
]

# Singleton registry — keyed by job_id
_jobs: dict[str, "JobState"] = {}


@dataclass
class StepInfo:
    name: str
    label: str
    state: str = "pending"   # pending | active | done | skipped | error
    duration_s: float = 0.0


@dataclass
class JobState:
    id: str
    video_path: str
    status: str = "running"  # running | done | error
    steps: list[StepInfo] = field(default_factory=list)
    progress: float = 0.0
    events: queue.Queue = field(default_factory=queue.Queue)
    error_msg: str = ""
    total_duration_s: float = 0.0

    @property
    def video_name(self) -> str:
        """Display basename of the source video.

        Computed in Python so the template needs no custom Jinja
        filter. Backslashes are normalized to ``/`` before splitting so
        a Windows-style path (``C:\\archive\\film.mp4``) yields the bare
        filename even when the server runs on POSIX — ``pathlib.Path``
        alone would not treat ``\\`` as a separator on Linux.
        """
        return PurePosixPath(self.video_path.replace("\\", "/")).name


def get_job(job_id: str) -> Optional[JobState]:
    return _jobs.get(job_id)


def active_jobs() -> list[JobState]:
    return [j for j in _jobs.values() if j.status == "running"]


def start_job(video_path: str, enabled_steps: set[str], cfg) -> str:
    """Create a job and start the pipeline in a background thread."""
    job_id = uuid.uuid4().hex[:8]
    job = JobState(
        id=job_id,
        video_path=video_path,
        steps=[
            StepInfo(name=name, label=label)
            for name, label in STEP_DEFS
        ],
    )
    _jobs[job_id] = job
    threading.Thread(
        target=_run_pipeline,
        args=(job, cfg, enabled_steps),
        daemon=True,
        name=f"pipeline-{job_id}",
    ).start()
    logger.info("Job %s started for %s", job_id, video_path)
    return job_id


# ── Pipeline runner (runs in a daemon thread) ─────────────────────────────────

def _run_pipeline(job: JobState, cfg, enabled_steps: set[str]) -> None:
    from cinemateca.pipeline import CatalogPipeline

    t_start = time.time()
    pipeline = CatalogPipeline(cfg)
    video_path = Path(job.video_path)

    keyframes_dir = Path(cfg.paths.frames_dir) / "scenes" / "keyframes_content"
    metadata_path = Path(cfg.paths.metadata_dir) / "keyframes_metadata.json"

    # Closures so keyframes_dir can be updated after scene_detection
    def run_step(step: StepInfo):
        nonlocal keyframes_dir
        match step.name:
            case "frame_extraction":
                return pipeline._step_frame_extraction(video_path)
            case "scene_detection":
                result = pipeline._step_scene_detection(video_path)
                if result.success and not result.skipped and isinstance(result.output, dict):
                    kd = result.output.get("keyframes_dir")
                    if kd:
                        keyframes_dir = Path(kd)
                return result
            case "visual_analysis":
                return pipeline._step_visual_analysis(keyframes_dir)
            case "embeddings":
                return pipeline._step_embeddings(metadata_path)
            case "llm_description":
                return pipeline._step_llm_description(metadata_path)

    for step in job.steps:
        if step.name not in enabled_steps:
            step.state = "skipped"
            job.events.put("update")
            continue

        step.state = "active"
        job.events.put("update")

        try:
            result = run_step(step)
        except Exception as exc:
            step.state = "error"
            job.error_msg = str(exc)
            job.status = "error"
            job.events.put("error")
            logger.exception("Job %s crashed at step %s", job.id, step.name)
            return

        step.duration_s = result.duration_s
        if result.skipped:
            step.state = "skipped"
        elif result.success:
            step.state = "done"
        else:
            step.state = "error"
            job.error_msg = result.error or ""

        done_count = sum(1 for s in job.steps if s.state in ("done", "skipped", "error"))
        job.progress = done_count / len(job.steps)
        job.events.put("update")

        if step.state == "error" and getattr(cfg.pipeline, "stop_on_error", False):
            job.status = "error"
            job.events.put("error")
            return

    job.status = "done" if all(s.state != "error" for s in job.steps) else "error"
    job.progress = 1.0
    job.total_duration_s = time.time() - t_start
    job.events.put("done")
    logger.info("Job %s finished — %.1fs, status=%s", job.id, job.total_duration_s, job.status)
