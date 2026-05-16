"""In-memory job registry and pipeline runner for the Processing tab.

Phase 4 hardens this module:

  * the runner no longer reaches into ``CatalogPipeline._step_*`` —
    it drives the public :meth:`CatalogPipeline.run_steps` API, which
    owns step selection, the dependency graph and input gating;
  * a small :class:`JobRegistry` replaces the bare module-global dict:
    it serializes all access behind a ``threading.Lock``, enforces a
    bounded retention cap on terminal jobs, and enforces the
    concurrency policy;
  * explicit lifecycle states ``created`` / ``running`` / ``done`` /
    ``error`` / ``cancelled`` plus cooperative cancellation (a flag the
    runner polls between steps);
  * dependency-aware gating surfaces ``blocked`` steps so an upstream
    failure can no longer silently combine stale mixed outputs.

Concurrency policy
------------------
**Single global active job.** This is an offline single-user tool; the
pipeline saturates CPU/GPU and (pre-Phase-5) every film shares one flat
data directory, so running two pipelines at once would thrash and risk
interleaved writes. A second start while a job is running is *rejected*
with a clear message rather than queued — the simplest correct policy
for this tool. (Phase 5's per-film data model may revisit this; the
policy is centralized in :meth:`JobRegistry.start` so that change is
local.)

Retention
---------
At most :data:`MAX_RETAINED_TERMINAL_JOBS` terminal jobs are kept; the
oldest are evicted (a single-user tool only needs recent history). The
active job is never evicted.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Optional

logger = logging.getLogger(__name__)

STEP_DEFS: list[tuple[str, str]] = [
    ("frame_extraction", "Frames"),
    ("scene_detection",  "Cenas"),
    ("visual_analysis",  "Visual"),
    ("embeddings",       "Embeddings"),
    ("llm_description",  "Descrições"),
]

# Lifecycle states a job can be in. ``created`` is transient (set before
# the worker thread starts); the worker flips it to ``running``.
STATUS_CREATED = "created"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_ERROR = "error"
STATUS_CANCELLED = "cancelled"
_TERMINAL = (STATUS_DONE, STATUS_ERROR, STATUS_CANCELLED)

# Bounded retention: keep at most this many terminal jobs in the
# registry; evict oldest first. The active job is never counted/evicted.
MAX_RETAINED_TERMINAL_JOBS = 20


@dataclass
class StepInfo:
    name: str
    label: str
    # pending | active | done | skipped | error | blocked
    state: str = "pending"
    duration_s: float = 0.0
    detail: str = ""


@dataclass
class JobState:
    id: str
    video_path: str
    status: str = STATUS_RUNNING  # created|running|done|error|cancelled
    steps: list[StepInfo] = field(default_factory=list)
    progress: float = 0.0
    events: queue.Queue = field(default_factory=queue.Queue)
    error_msg: str = ""
    total_duration_s: float = 0.0
    created_at: float = field(default_factory=time.time)
    # Cooperative-cancel flag; the runner polls it between steps.
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

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

    @property
    def is_terminal(self) -> bool:
        return self.status in _TERMINAL

    def request_cancel(self) -> None:
        """Signal cooperative cancellation (idempotent)."""
        self._cancel.set()

    @property
    def cancel_requested(self) -> bool:
        return self._cancel.is_set()


class ConcurrencyRejected(Exception):
    """Raised by :meth:`JobRegistry.start` when a job is already active.

    Carries the in-flight job so the caller can surface a useful message.
    """

    def __init__(self, active: JobState):
        self.active = active
        super().__init__(
            f"A job is already running ({active.video_name}); "
            f"this single-user tool runs one pipeline at a time."
        )


class JobRegistry:
    """Thread-safe job registry with bounded retention.

    All mutation/read of the underlying dict happens under ``_lock`` so
    the SSE stream thread, the worker threads and request handlers never
    observe a torn state.
    """

    def __init__(self, max_terminal: int = MAX_RETAINED_TERMINAL_JOBS):
        self._jobs: dict[str, JobState] = {}
        self._lock = threading.Lock()
        self._max_terminal = max_terminal

    # ── Reads ─────────────────────────────────────────────────────────
    def get(self, job_id: str) -> Optional[JobState]:
        with self._lock:
            return self._jobs.get(job_id)

    def active(self) -> list[JobState]:
        with self._lock:
            return [
                j for j in self._jobs.values() if j.status == STATUS_RUNNING
            ]

    def _active_locked(self) -> Optional[JobState]:
        for j in self._jobs.values():
            if j.status == STATUS_RUNNING:
                return j
        return None

    def all(self) -> list[JobState]:
        with self._lock:
            return list(self._jobs.values())

    # ── Mutation ──────────────────────────────────────────────────────
    def _evict_locked(self) -> None:
        """Drop oldest terminal jobs beyond the retention cap."""
        terminal = sorted(
            (j for j in self._jobs.values() if j.is_terminal),
            key=lambda j: j.created_at,
        )
        excess = len(terminal) - self._max_terminal
        for j in terminal[:max(0, excess)]:
            self._jobs.pop(j.id, None)
            logger.debug("Evicted retained job %s", j.id)

    def start(self, video_path: str, enabled_steps: set[str], cfg) -> str:
        """Register a job and launch its worker thread.

        Enforces the single-global-active-job policy: raises
        :class:`ConcurrencyRejected` if a job is already running.
        """
        with self._lock:
            existing = self._active_locked()
            if existing is not None:
                raise ConcurrencyRejected(existing)

            job_id = uuid.uuid4().hex[:8]
            job = JobState(
                id=job_id,
                video_path=video_path,
                status=STATUS_CREATED,
                steps=[
                    StepInfo(name=name, label=label)
                    for name, label in STEP_DEFS
                ],
            )
            self._jobs[job_id] = job
            self._evict_locked()

        threading.Thread(
            target=_run_pipeline,
            args=(job, cfg, enabled_steps),
            daemon=True,
            name=f"pipeline-{job_id}",
        ).start()
        logger.info("Job %s started for %s", job_id, video_path)
        return job_id

    def prune(self) -> None:
        """Evict oldest terminal jobs beyond the cap (thread-safe).

        Called by the worker thread once a job reaches a terminal state
        so retention is enforced promptly, not only at the next
        ``start()``.
        """
        with self._lock:
            self._evict_locked()

    def cancel(self, job_id: str) -> bool:
        """Request cooperative cancellation of a job.

        Returns True if the job exists and is still running (a cancel
        was signalled); False otherwise.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.is_terminal:
                return False
        job.request_cancel()
        logger.info("Cancellation requested for job %s", job_id)
        return True


# Process-global registry instance. ``conftest.py`` and ``test_sse.py``
# reset hermeticity via ``monkeypatch.setattr(jobs, "_jobs", {})``; the
# module-level ``_jobs`` name below is kept as a thin alias bound to the
# live registry's dict so that existing reset pattern keeps working
# without touching every test.
_registry = JobRegistry()
_jobs: dict[str, JobState] = _registry._jobs


def _sync_registry_dict() -> None:
    """Re-point the registry at the current module-level ``_jobs``.

    Tests do ``monkeypatch.setattr(jobs, "_jobs", {})`` to isolate the
    registry. That rebinds the module name but not the registry's
    internal dict, so the public helpers must read whatever ``_jobs``
    currently is. Calling this at the top of each public function keeps
    the registry and the (possibly monkeypatched) module global in sync
    without changing the long-standing test reset idiom.
    """
    if _registry._jobs is not _jobs:
        _registry._jobs = _jobs


def get_job(job_id: str) -> Optional[JobState]:
    _sync_registry_dict()
    return _registry.get(job_id)


def active_jobs() -> list[JobState]:
    _sync_registry_dict()
    return _registry.active()


def cancel_job(job_id: str) -> bool:
    _sync_registry_dict()
    return _registry.cancel(job_id)


def _prune_registry() -> None:
    _sync_registry_dict()
    _registry.prune()


def start_job(video_path: str, enabled_steps: set[str], cfg) -> str:
    """Create a job and start the pipeline in a background thread.

    Raises :class:`ConcurrencyRejected` if a job is already running
    (single-global-active-job policy).
    """
    _sync_registry_dict()
    return _registry.start(video_path, enabled_steps, cfg)


# ── Pipeline runner (runs in a daemon thread) ─────────────────────────────────

def _run_pipeline(job: JobState, cfg, enabled_steps: set[str]) -> None:
    """Drive :meth:`CatalogPipeline.run_steps` and mirror state onto the job.

    The runner no longer knows the dependency graph or calls private
    ``_step_*``; ``run_steps`` owns selection + gating and reports each
    step's lifecycle through ``progress_cb``. Cancellation is
    cooperative: ``run_steps`` polls ``cancel_check`` between steps and
    raises ``StepCancelled`` when the job's cancel flag is set.
    """
    from cinemateca.pipeline import CatalogPipeline, StepCancelled

    t_start = time.time()
    job.status = STATUS_RUNNING
    pipeline = CatalogPipeline(cfg)

    by_name: dict[str, StepInfo] = {s.name: s for s in job.steps}

    # Steps not selected are immediately marked skipped (parity with the
    # old runner, which skipped unselected steps before the loop body).
    for s in job.steps:
        if s.name not in enabled_steps:
            s.state = "skipped"
    if any(s.state == "skipped" for s in job.steps):
        job.events.put("update")

    def _recompute_progress() -> None:
        done = sum(
            1
            for s in job.steps
            if s.state in ("done", "skipped", "error", "blocked")
        )
        job.progress = done / len(job.steps) if job.steps else 1.0

    def progress_cb(name: str, phase: str, run) -> None:
        step = by_name.get(name)
        if step is None:
            return
        if phase == "start":
            step.state = "active"
            job.events.put("update")
            return
        # phase == "finish"
        step.state = run.state
        step.duration_s = run.duration_s
        if run.state == "error":
            step.detail = run.error or ""
            if not job.error_msg:
                job.error_msg = run.error or ""
        elif run.state == "blocked":
            step.detail = run.error or ""
        _recompute_progress()
        job.events.put("update")

    try:
        results = pipeline.run_steps(
            job.video_path,
            steps=[name for name, _ in STEP_DEFS if name in enabled_steps],
            progress_cb=progress_cb,
            cancel_check=lambda: job.cancel_requested,
        )
    except StepCancelled:
        job.status = STATUS_CANCELLED
        _recompute_progress()
        job.total_duration_s = time.time() - t_start
        if not job.error_msg:
            job.error_msg = "Cancelled by user."
        job.events.put("cancelled")
        logger.info("Job %s cancelled after %.1fs", job.id, job.total_duration_s)
        _prune_registry()
        return
    except Exception as exc:  # defensive: run_steps wraps step errors,
        # so this only fires on an orchestration-level fault.
        job.error_msg = str(exc)
        job.status = STATUS_ERROR
        _recompute_progress()
        job.total_duration_s = time.time() - t_start
        job.events.put("error")
        logger.exception("Job %s crashed in run_steps", job.id)
        _prune_registry()
        return

    had_error = any(
        r.state in ("error", "blocked") for r in results.runs
    )
    job.status = STATUS_ERROR if had_error else STATUS_DONE
    job.progress = 1.0
    job.total_duration_s = time.time() - t_start
    job.events.put("error" if had_error else "done")
    logger.info(
        "Job %s finished — %.1fs, status=%s",
        job.id,
        job.total_duration_s,
        job.status,
    )
    _prune_registry()
