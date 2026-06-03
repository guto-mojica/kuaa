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

WS-2 A9 — explicit guarded state machine
-----------------------------------------
Job status is now :class:`JobStatus` (a ``str`` enum) with an explicit
transition table :data:`_TRANSITIONS`. :meth:`JobState.transition_to`
enforces legality, raising :exc:`cinemateca.errors.PipelineError` on
an illegal flip.  The module-level ``STATUS_*`` names are kept as enum
aliases so every existing import site (``api/routes/processing.py``,
tests, the runner itself) continues to work without change.

Event-bus scope (§16)
---------------------
:class:`EventBroadcaster` is intentionally the **sole** event surface.
Multi-job concurrency and a richer publish/subscribe bus are **out of
scope** for this single-user offline tool (spec §16).  Do NOT expand
the broadcaster into a multi-job fan-out: that work belongs in a future
version when per-film data isolation makes concurrent pipelines safe.
Deleting the broadcaster would regress the SSE multi-tab replay tested
in ``tests/test_jobs_broadcaster.py`` and
``tests/test_processing_log_replay.py``.

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

import contextlib
import logging
import queue
import threading
import time
import uuid
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from typing import Any

logger = logging.getLogger(__name__)

STEP_DEFS: list[tuple[str, str]] = [
    ("frame_extraction", "Frames"),
    ("scene_detection", "Cenas"),
    ("visual_analysis", "Visual"),
    ("embeddings", "Embeddings"),
    ("llm_description", "Descrições"),
]


class JobStatus(str, Enum):
    """Explicit lifecycle states for a pipeline job.

    ``str`` mixin means every member compares equal to its string value
    so existing code that checks ``job.status == "running"`` or
    ``job.status in ("done", "error", "cancelled")`` keeps working
    without modification.
    """

    CREATED = "created"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


# Module-level aliases — kept for every existing import site.
# ``from api.jobs import STATUS_RUNNING`` still works, and the values
# compare equal to the plain strings they replaced.
STATUS_CREATED: JobStatus = JobStatus.CREATED
STATUS_RUNNING: JobStatus = JobStatus.RUNNING
STATUS_DONE: JobStatus = JobStatus.DONE
STATUS_ERROR: JobStatus = JobStatus.ERROR
STATUS_CANCELLED: JobStatus = JobStatus.CANCELLED

# Legal transitions: maps a status to the set of statuses it may move into.
# Terminal states (DONE / ERROR / CANCELLED) map to the empty frozenset —
# they are frozen.  Enforced by :meth:`JobState.transition_to`.
_TRANSITIONS: dict[JobStatus, frozenset[JobStatus]] = {
    JobStatus.CREATED: frozenset({JobStatus.RUNNING, JobStatus.CANCELLED}),
    JobStatus.RUNNING: frozenset({JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED}),
    JobStatus.DONE: frozenset(),
    JobStatus.ERROR: frozenset(),
    JobStatus.CANCELLED: frozenset(),
}

_ACTIVE = (JobStatus.CREATED, JobStatus.RUNNING)
_TERMINAL = (JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED)

# Bounded retention: keep at most this many terminal jobs in the
# registry; evict oldest first. The active job is never counted/evicted.
MAX_RETAINED_TERMINAL_JOBS = 20

# Per-job log ring buffer. A 30-minute pipeline at typical verbosity
# produces ~200 lines; 500 leaves comfortable headroom and bounds RAM
# at ~100KB per job (a log row is ~200B serialised). Survives page
# navigation because it lives on JobState in the in-memory registry.
LOG_BUFFER_MAXLEN = 500

# Per-subscriber SSE event queue size. Large enough that a brief
# network back-pressure stall (a few seconds) doesn't drop events; if
# a consumer is genuinely dead we'd rather drop on its side than block
# the producer thread. The replay path on reconnect catches it up from
# the JobState log buffer.
SUBSCRIBER_QUEUE_MAXLEN = 500


# ── Pub/sub event broadcaster ─────────────────────────────────────────────────


# An event is a (name, data) tuple. ``name`` is one of:
#   * ``"update"``      — stepper progress signal (data: None)
#   * ``"log"``         — a captured pipeline log row (data: dict with
#                         ``t`` / ``lv`` / ``m`` keys; matches the
#                         ``processing_log_line.html`` template shape)
#   * ``"done"`` / ``"error"`` / ``"cancelled"`` — terminal signals
#                         (data: None). The SSE generator renders the
#                         final stepper at emit time.
Event = tuple[str, Any]


class EventBroadcaster:
    """Fan-out events to all currently-subscribed consumer queues.

    Replaces the old single-consumer ``queue.Queue`` so multiple SSE
    streams (two browser tabs, reloads, the log pane + the stepper
    div) can all consume the same job's progress without racing each
    other for events.

    Producer side (called from the pipeline worker thread) uses
    :meth:`publish`; consumers (the SSE generator) get a per-connection
    queue via :meth:`subscribe` and MUST call :meth:`unsubscribe` when
    their connection closes so a dead client doesn't leak a queue for
    the rest of the job.

    Slow-consumer policy: if a subscriber's queue is full, the event
    is silently dropped FOR THAT SUBSCRIBER ONLY — the producer never
    blocks, and other subscribers keep flowing. A reconnecting client
    catches up from the JobState log buffer (which is the durable
    layer); the broadcaster itself is the live wire only.
    """

    def __init__(self) -> None:
        self._subscribers: list[queue.Queue] = []
        self._lock = threading.Lock()

    def subscribe(self, maxsize: int = SUBSCRIBER_QUEUE_MAXLEN) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=maxsize)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass  # already unsubscribed; idempotent

    def publish(self, event: Event) -> None:
        # Snapshot the subscriber list under the lock so a concurrent
        # subscribe/unsubscribe can't mutate it while we iterate.
        with self._lock:
            subs = list(self._subscribers)
        for q in subs:
            try:
                q.put_nowait(event)
            except queue.Full:
                # Slow consumer: drop the event for this subscriber.
                # See class docstring for rationale.
                pass

    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)


# ── Pipeline log capture ──────────────────────────────────────────────────────


# Logger namespaces whose output we want to surface in the Processing
# UI. ``cinemateca.*`` is the entire AI core; ``api.jobs`` is this
# module itself (so the runner's "step start / step finish" lines
# also surface). Tightly scoped so unrelated noise (httpx, uvicorn,
# asyncio) never lands in the user-visible log pane.
_PIPELINE_LOGGER_NAMES = ("cinemateca", "api.jobs")

# Python logging levelno → template ``lv`` code. The template
# (processing_log_line.html + proc.css) ships rules for i / d / w / s
# (success). 'e' is added for error/critical so a future CSS rule can
# style it distinctly; today it falls through to the default span
# styling which is still legible.
_LEVEL_LV = {
    logging.DEBUG: "d",
    logging.INFO: "i",
    logging.WARNING: "w",
    logging.ERROR: "e",
    logging.CRITICAL: "e",
}


class _JobLogHandler(logging.Handler):
    """A logging.Handler that routes captured records onto a
    :class:`JobState` (durable ring buffer + live broadcast).

    Construct via :func:`install_pipeline_log_handler` (the
    contextmanager that attaches/detaches it).
    """

    def __init__(self, job: JobState) -> None:
        super().__init__(level=logging.DEBUG)
        self._job = job

    def emit(self, record: logging.LogRecord) -> None:
        try:
            row = {
                "t": time.strftime("%H:%M:%S", time.localtime(record.created)),
                "lv": _LEVEL_LV.get(record.levelno, "i"),
                "m": record.getMessage(),
            }
        except Exception:  # noqa: BLE001 — handler MUST NOT raise
            return
        # Buffer first (durable layer) then broadcast (live layer) so a
        # consumer that subscribes between the two operations still
        # sees the row on replay.
        self._job.log.append(row)
        try:
            self._job.publish("log", row)
        except Exception:  # noqa: BLE001 — broadcaster failure must
            # not crash a producer logger (extremely defensive; the
            # broadcaster's publish is itself try/except internally).
            pass


@contextlib.contextmanager
def install_pipeline_log_handler(job: JobState) -> Iterator[_JobLogHandler]:
    """Context manager that attaches a :class:`_JobLogHandler` to the
    pipeline logger namespaces for the duration of the ``with`` block.

    We also temporarily lower each target logger's ``level`` to DEBUG
    so INFO/DEBUG records actually reach the handler (the project
    default is INFO; in a fresh test process the root WARNING level
    would otherwise drop everything below WARNING before our handler
    sees it). The original levels are restored in ``finally`` so a
    crash inside the runner cannot leak verbose logging across the
    rest of the server's lifetime.
    """
    handler = _JobLogHandler(job)
    attached = [logging.getLogger(name) for name in _PIPELINE_LOGGER_NAMES]
    saved_levels = [lg.level for lg in attached]
    for lg in attached:
        lg.addHandler(handler)
        if lg.level == logging.NOTSET or lg.level > logging.DEBUG:
            lg.setLevel(logging.DEBUG)
    try:
        yield handler
    finally:
        for lg, original in zip(attached, saved_levels):
            try:
                lg.removeHandler(handler)
            except Exception:  # noqa: BLE001 — removal MUST be idempotent
                pass
            try:
                lg.setLevel(original)
            except Exception:  # noqa: BLE001 — defensive: never raise
                pass


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
    status: JobStatus = JobStatus.RUNNING  # created|running|done|error|cancelled
    steps: list[StepInfo] = field(default_factory=list)
    progress: float = 0.0
    # Multi-consumer event bus. Replaces the old single-consumer
    # ``queue.Queue`` so multiple SSE streams to the same job (two
    # tabs, reload-during-run, the log pane + the stepper div) all
    # see the same events without racing.
    broadcaster: EventBroadcaster = field(default_factory=EventBroadcaster, repr=False)
    # Bounded ring buffer of captured pipeline log lines. Each row:
    # ``{"t": "HH:MM:SS", "lv": "i|d|w|e|s", "m": "message text"}``
    # matching ``processing_log_line.html``. This is the durable
    # layer that lets a user navigate away and return to a full
    # replay of what the pipeline did while they were gone.
    log: deque = field(
        default_factory=lambda: deque(maxlen=LOG_BUFFER_MAXLEN),
        repr=False,
    )
    error_msg: str = ""
    total_duration_s: float = 0.0
    created_at: float = field(default_factory=time.time)
    # Cooperative-cancel flag; the runner polls it between steps.
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    # ── Pub/sub convenience methods ──────────────────────────────────
    def publish(self, name: str, data: Any = None) -> None:
        """Publish ``(name, data)`` to every live subscriber.

        Convenience wrapper so producer code reads as
        ``job.publish("update")`` / ``job.publish("log", row)`` rather
        than reaching into ``job.broadcaster`` directly.
        """
        self.broadcaster.publish((name, data))

    def subscribe(self, maxsize: int = SUBSCRIBER_QUEUE_MAXLEN) -> queue.Queue:
        return self.broadcaster.subscribe(maxsize=maxsize)

    def unsubscribe(self, q: queue.Queue) -> None:
        self.broadcaster.unsubscribe(q)

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

    def transition_to(self, new: JobStatus) -> None:
        """Move to *new* status, raising :exc:`~cinemateca.errors.PipelineError`
        if the transition is not in :data:`_TRANSITIONS`.

        This is the authoritative write path for job lifecycle changes.
        Direct ``job.status = ...`` assignments in test fixtures are still
        valid (they bypass the guard intentionally — fixtures build
        pre-configured job states directly).
        """
        from cinemateca.errors import PipelineError

        allowed = _TRANSITIONS.get(self.status, frozenset())
        if new not in allowed:
            raise PipelineError(
                f"illegal job transition {self.status.value}->{new.value}",
                code="pipeline.illegal_transition",
            )
        self.status = new


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


@dataclass
class PendingEntry:
    """A job waiting in the batch queue, not yet started."""

    id: str
    video_path: str
    steps: frozenset
    cfg: Any = field(default=None, repr=False)
    created_at: float = field(default_factory=time.time)

    @property
    def video_name(self) -> str:
        return PurePosixPath(self.video_path.replace("\\", "/")).name


class JobRegistry:
    """Thread-safe job registry with bounded retention.

    ``_lock`` serializes all mutation/read of the ``_jobs`` *container*,
    so lookups and the retention sweep never race with insert/evict. Each
    job additionally carries its own ``queue.Queue`` event channel, which
    is itself thread-safe; the SSE stream is driven by that queue, not by
    polling :class:`JobState` fields.

    Scope of the guarantee: this is *container* consistency plus a
    thread-safe per-job event channel. It is NOT a claim of atomic
    multi-field snapshots of a :class:`JobState` — the worker thread
    writes individual fields (``status`` / ``steps`` / ``progress`` /
    ``error_msg``) without a per-job lock, and request handlers read them
    lock-free, so a concurrent multi-field read may observe a partially
    applied progress update. This is acceptable here because SSE
    consumers react to discrete queue events rather than to a sampled
    field tuple.
    """

    def __init__(self, max_terminal: int = MAX_RETAINED_TERMINAL_JOBS):
        self._jobs: dict[str, JobState] = {}
        self._pending: list[PendingEntry] = []
        self._lock = threading.Lock()
        self._max_terminal = max_terminal

    # ── Reads ─────────────────────────────────────────────────────────
    def get(self, job_id: str) -> JobState | None:
        with self._lock:
            return self._jobs.get(job_id)

    def active(self) -> list[JobState]:
        with self._lock:
            return [j for j in self._jobs.values() if j.status in _ACTIVE]

    def _active_locked(self) -> JobState | None:
        for j in self._jobs.values():
            if j.status in _ACTIVE:
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
        for j in terminal[: max(0, excess)]:
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
                steps=[StepInfo(name=name, label=label) for name, label in STEP_DEFS],
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

    # ── Test isolation ────────────────────────────────────────────────
    def reset(self) -> None:
        """Drop every tracked job, under the registry lock.

        Test-only hermeticity hook: each test gets a pristine registry
        (no leaked ``JobState`` from a prior test). Clearing happens
        behind ``_lock`` so it can never race a concurrent insert/evict
        from a still-running worker thread, and the ``_jobs`` *container*
        identity is preserved (``.clear()`` not reassignment) so any
        alias still observes the empty dict.
        """
        with self._lock:
            self._jobs.clear()
            self._pending.clear()

    def add(self, job: JobState) -> None:
        """Insert a pre-built :class:`JobState`, under the lock.

        Test-only hook: tests that need a job in a specific state
        (e.g. a running job for SSE/processing assertions) construct
        it directly and register it here, rather than going through
        :meth:`start` which spawns a real worker thread. Lock-guarded
        for the same race-freedom as the rest of the registry.
        """
        with self._lock:
            self._jobs[job.id] = job

    # ── Batch queue ───────────────────────────────────────────────────────

    def enqueue(self, video_path: str, steps: set[str], cfg) -> str:
        """Start immediately if idle, otherwise add to the pending queue.

        Returns the job_id (if started) or pending entry id (if queued).
        Never raises :class:`ConcurrencyRejected`.
        """
        with self._lock:
            if self._active_locked() is not None:
                entry_id = uuid.uuid4().hex[:8]
                entry = PendingEntry(
                    id=entry_id,
                    video_path=video_path,
                    steps=frozenset(steps),
                    cfg=cfg,
                )
                self._pending.append(entry)
                logger.info(
                    "Enqueued pending job %s for %s (depth=%d)",
                    entry_id, video_path, len(self._pending),
                )
                return entry_id
        # No active job — start immediately (takes its own lock).
        return self.start(video_path, steps, cfg)

    def queue_only(self, video_path: str, steps: set[str], cfg) -> str:
        """Always add to the pending queue, even when the registry is idle.

        Use this to pre-populate the queue before explicitly starting it
        with :meth:`start_queue`. Never raises :class:`ConcurrencyRejected`.
        """
        with self._lock:
            entry_id = uuid.uuid4().hex[:8]
            entry = PendingEntry(
                id=entry_id,
                video_path=video_path,
                steps=frozenset(steps),
                cfg=cfg,
            )
            self._pending.append(entry)
            logger.info(
                "Queued (pending) job %s for %s (depth=%d)",
                entry_id, video_path, len(self._pending),
            )
            return entry_id

    def start_queue(self) -> str | None:
        """Pop the first pending entry and start it if the registry is idle.

        Returns the new job_id, or ``None`` if the registry is busy or the
        queue is empty.
        """
        with self._lock:
            if not self._pending or self._active_locked() is not None:
                return None
            entry = self._pending.pop(0)

        logger.info("Starting queued job %s for %s", entry.id, entry.video_path)
        try:
            return self.start(entry.video_path, set(entry.steps), entry.cfg)
        except ConcurrencyRejected:
            with self._lock:
                self._pending.insert(0, entry)
            logger.warning("start_queue: concurrency conflict, re-queued %s", entry.id)
            return None

    def pending_entries(self) -> list[PendingEntry]:
        """Snapshot of the pending queue, oldest first."""
        with self._lock:
            return list(self._pending)

    def remove_pending(self, entry_id: str) -> bool:
        """Remove a pending entry by id. Returns True if found and removed."""
        with self._lock:
            for i, entry in enumerate(self._pending):
                if entry.id == entry_id:
                    del self._pending[i]
                    logger.info("Removed pending job %s", entry_id)
                    return True
        return False

    def drain_pending(self) -> None:
        """Pop the next pending entry and start it if the registry is now idle.

        Called from the runner thread at the end of each terminal path so
        the batch queue auto-advances without any external polling.
        """
        with self._lock:
            if not self._pending or self._active_locked() is not None:
                return
            entry = self._pending.pop(0)

        logger.info("Draining queue — starting %s for %s", entry.id, entry.video_path)
        try:
            self.start(entry.video_path, set(entry.steps), entry.cfg)
        except ConcurrencyRejected:
            # Another thread beat us; put it back at the front.
            with self._lock:
                self._pending.insert(0, entry)
            logger.warning("drain_pending: concurrency conflict, re-queued %s", entry.id)


# Process-global registry instance. Tests isolate state by calling
# ``jobs._registry.reset()`` (a lock-guarded clear) and seed jobs via
# ``jobs._registry.add(job)`` — see ``tests/conftest.py``.
_registry = JobRegistry()


def get_all_jobs() -> list:
    """Return a snapshot of all jobs (active + terminal) from the registry."""
    return _registry.all()


def get_job(job_id: str) -> JobState | None:
    return _registry.get(job_id)


def active_jobs() -> list[JobState]:
    return _registry.active()


def all_jobs() -> list[JobState]:
    """Return every job in the registry (active, completed, cancelled, errored).

    Wraps ``_registry.all()`` as a public function so routes and services can
    retrieve the full job history without reaching into the private registry
    object.  Use :func:`active_jobs` when you only care about running jobs.
    """
    return _registry.all()


def cancel_job(job_id: str) -> bool:
    return _registry.cancel(job_id)


def _prune_registry() -> None:
    _registry.prune()


def start_job(video_path: str, enabled_steps: set[str], cfg) -> str:
    """Create a job and start the pipeline in a background thread.

    Raises :class:`ConcurrencyRejected` if a job is already running
    (single-global-active-job policy).
    """
    return _registry.start(video_path, enabled_steps, cfg)


def enqueue_job(video_path: str, enabled_steps: set[str], cfg) -> str:
    """Start immediately if idle, or add to the pending batch queue.

    Unlike :func:`start_job`, never raises :class:`ConcurrencyRejected`.
    Returns the job id (started) or pending entry id (queued).
    """
    return _registry.enqueue(video_path, enabled_steps, cfg)


def queue_job(video_path: str, enabled_steps: set[str], cfg) -> str:
    """Always add to the pending queue without starting (Option B gesture).

    Lets the user pre-populate the queue before explicitly starting it
    with :func:`start_queued_jobs`.
    """
    return _registry.queue_only(video_path, enabled_steps, cfg)


def start_queued_jobs() -> str | None:
    """Start the first pending entry if the registry is idle.

    Returns the new job_id, or None if busy / queue empty.
    """
    return _registry.start_queue()


def pending_jobs() -> list[PendingEntry]:
    """Snapshot of the pending batch queue (oldest first)."""
    return _registry.pending_entries()


def remove_pending_job(entry_id: str) -> bool:
    """Remove a pending entry by id. Returns True if found."""
    return _registry.remove_pending(entry_id)


# ── Pipeline runner (runs in a daemon thread) ─────────────────────────────────


def _slug_for_video(video_path: str, cfg) -> str:
    """Return the registry slug that owns *video_path*.

    Matches by ``raw_filename`` in ``films.json`` first — reliable even
    when the filename stem differs from the slug (e.g. raw file is
    ``jeca_tatu_1959.mp4`` but slug is ``jeca_tatu``).  Falls back to
    the stem-based heuristic only when no registry entry matches.
    """
    from pathlib import Path, PurePosixPath

    from cinemateca.library import load_registry

    filename = PurePosixPath(video_path.replace("\\", "/")).name
    try:
        library_dir = Path(getattr(getattr(cfg, "paths", None), "library_dir", ""))
        if library_dir.name:
            for slug, entry in load_registry(library_dir).items():
                if entry.get("raw_filename", "") == filename:
                    return slug
    except Exception as exc:
        logger.warning("Could not read registry for slug lookup: %s", exc)
    # Fallback: derive from stem (may not match registry slug)
    stem = PurePosixPath(video_path.replace("\\", "/")).stem
    return stem.lower().replace(" ", "_")


def _run_pipeline(job: JobState, cfg, enabled_steps: set[str]) -> None:
    """Drive :meth:`CatalogPipeline.run_steps` and mirror state onto the job.

    The runner no longer knows the dependency graph or calls private
    ``_step_*``; ``run_steps`` owns selection + gating and reports each
    step's lifecycle through ``progress_cb``. Cancellation is
    cooperative: ``run_steps`` polls ``cancel_check`` between steps and
    raises ``StepCancelled`` when the job's cancel flag is set.
    """

    from cinemateca.library import FilmContext
    from cinemateca.pipeline import CatalogPipeline, StepCancelled, StepResults, StepRun
    from cinemateca.run_manifest import write_run_manifest

    t_start = time.time()
    job.transition_to(JobStatus.RUNNING)

    slug = _slug_for_video(job.video_path, cfg)
    logger.info(
        "[job=%s] runner entered — video=%s slug=%s steps=%s",
        job.id,
        job.video_path,
        slug,
        sorted(enabled_steps),
    )
    ctx = FilmContext.for_film(cfg, slug)
    ctx.metadata_dir.mkdir(parents=True, exist_ok=True)
    ctx.frames_dir.mkdir(parents=True, exist_ok=True)
    ctx.embeddings_dir.mkdir(parents=True, exist_ok=True)

    pipeline = CatalogPipeline(cfg, slug=slug)

    by_name: dict[str, StepInfo] = {s.name: s for s in job.steps}

    def _snapshot_job_steps() -> StepResults:
        runs = [
            StepRun(
                name=s.name,
                state=s.state,
                duration_s=s.duration_s,
                error=s.detail or None,
            )
            for s in job.steps
            if s.name in enabled_steps
        ]
        return StepResults(video_path=job.video_path, runs=runs)

    def _write_manifest(
        result=None, *, status: str | None = None, error: str | None = None
    ) -> None:
        try:
            write_run_manifest(
                cfg,
                job.video_path,
                result,
                status=status,
                started_at_epoch=t_start,
                error=error,
            )
        except Exception as exc:  # noqa: BLE001 - manifest must not mask job result
            logger.warning("Could not write run manifest for job %s: %s", job.id, exc)

    # Steps not selected are immediately marked skipped (parity with the
    # old runner, which skipped unselected steps before the loop body).
    for s in job.steps:
        if s.name not in enabled_steps:
            s.state = "skipped"
    if any(s.state == "skipped" for s in job.steps):
        job.publish("update")

    def _recompute_progress() -> None:
        done = sum(1 for s in job.steps if s.state in ("done", "skipped", "error", "blocked"))
        job.progress = done / len(job.steps) if job.steps else 1.0

    def progress_cb(name: str, phase: str, run) -> None:
        step = by_name.get(name)
        if step is None:
            return
        if phase == "start":
            step.state = "active"
            logger.info("[job=%s] step %s — start", job.id, name)
            job.publish("update")
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
        logger.info(
            "[job=%s] step %s — finish (state=%s, %.1fs)",
            job.id,
            name,
            run.state,
            run.duration_s,
        )
        job.publish("update")

    # Install the pipeline log handler so every record emitted by
    # cinemateca.* and api.jobs during this job's lifetime lands in
    # the durable ring buffer + broadcasts to live SSE consumers.
    # The outer try/finally guarantees the handler is removed even on
    # an early-return exception path, so the next job starts with a
    # clean logging tree.
    with install_pipeline_log_handler(job):
        try:
            results = pipeline.run_steps(
                job.video_path,
                steps=[name for name, _ in STEP_DEFS if name in enabled_steps],
                progress_cb=progress_cb,
                cancel_check=lambda: job.cancel_requested,
            )
        except StepCancelled:
            job.transition_to(JobStatus.CANCELLED)
            _recompute_progress()
            job.total_duration_s = time.time() - t_start
            if not job.error_msg:
                job.error_msg = "Cancelled by user."
            _write_manifest(
                _snapshot_job_steps(),
                status=STATUS_CANCELLED,
                error=job.error_msg,
            )
            job.publish("cancelled")
            logger.info("[job=%s] cancelled after %.1fs", job.id, job.total_duration_s)
            _prune_registry()
            _registry.drain_pending()
            return
        except Exception as exc:  # defensive: run_steps wraps step errors,
            # so this only fires on an orchestration-level fault.
            job.error_msg = str(exc)
            job.transition_to(JobStatus.ERROR)
            _recompute_progress()
            job.total_duration_s = time.time() - t_start
            _write_manifest(status=STATUS_ERROR, error=job.error_msg)
            job.publish("error")
            logger.exception("[job=%s] crashed in run_steps", job.id)
            _prune_registry()
            _registry.drain_pending()
            return

        had_error = any(r.state in ("error", "blocked") for r in results.runs)
        job.transition_to(JobStatus.ERROR if had_error else JobStatus.DONE)
        job.progress = 1.0
        job.total_duration_s = time.time() - t_start
        _write_manifest(
            results,
            status=job.status,
            error=job.error_msg or None,
        )
        job.publish("error" if had_error else "done")
        logger.info(
            "[job=%s] finished — %.1fs, status=%s",
            job.id,
            job.total_duration_s,
            job.status,
        )
        _prune_registry()
        _registry.drain_pending()
