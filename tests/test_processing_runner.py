"""
tests/test_processing_runner.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Phase 4: Processing runner refactor.

Pins the public ``CatalogPipeline.run_steps`` API + ``api.jobs``
registry contract:

  * public selected-step API delegates to the existing ``_step_*``
    (we stub those so NO real video/CLIP/torch/models run);
  * dependency-aware gating: when ``scene_detection`` fails its
    downstream consumers are ``blocked`` and produce NO output (the
    historical "stale mixed output" defect);
  * legitimate subset runs still work when prior artefacts exist on
    disk;
  * job lifecycle states + bounded retention + thread-safe registry +
    single-global-active-job concurrency policy + cooperative
    cancellation.

All hermetic: ``_step_*`` are monkeypatched onto a real pipeline; the
job-layer tests stub the pipeline class entirely.
"""

from __future__ import annotations

import threading
import time
import types

import pytest

from cinemateca.pipeline import (
    STEP_DEPS,
    STEP_ORDER,
    CatalogPipeline,
    StepCancelled,
    StepResult,
)

# ── Fake config + pipeline helpers ────────────────────────────────────────────


def _fake_cfg(tmp_path):
    """Minimal config object with just the paths run_steps touches."""
    paths = types.SimpleNamespace(
        frames_dir=tmp_path / "frames",
        metadata_dir=tmp_path / "meta",
    )
    (paths.frames_dir / "scenes" / "keyframes_content").mkdir(
        parents=True, exist_ok=True
    )
    paths.metadata_dir.mkdir(parents=True, exist_ok=True)
    return types.SimpleNamespace(paths=paths)


def _pipeline_with_stubbed_steps(tmp_path, *, outcomes: dict[str, str]):
    """A real CatalogPipeline whose ``_step_*`` are fakes.

    ``outcomes`` maps step name -> "ok" | "fail" | "skip". The fakes
    return real ``StepResult`` objects, so run_steps' orchestration is
    exercised without any model/video.
    """
    cfg = _fake_cfg(tmp_path)
    p = CatalogPipeline(cfg)
    calls: list[str] = []

    def make(name):
        def _step(*_a, **_k):
            calls.append(name)
            o = outcomes.get(name, "ok")
            if o == "fail":
                return StepResult(name=name, success=False, error=f"{name} boom")
            if o == "skip":
                return StepResult(name=name, success=True, skipped=True)
            out = None
            if name == "scene_detection":
                kfd = cfg.paths.frames_dir / "scenes" / "keyframes_content"
                out = {"keyframes_dir": kfd}
            return StepResult(
                name=name, success=True, duration_s=1.0, output=out
            )

        return _step

    p._step_frame_extraction = make("frame_extraction")
    p._step_scene_detection = make("scene_detection")
    p._step_visual_analysis = make("visual_analysis")
    p._step_embeddings = make("embeddings")
    p._step_llm_description = make("llm_description")
    return p, cfg, calls


# ── Dependency graph encoding ─────────────────────────────────────────────────


def test_dependency_graph_matches_verified_prereqs():
    """Roots have no deps; the keyframe-metadata consumers depend on
    scene_detection (verified against pipeline.run() + _step_*)."""
    assert STEP_DEPS["frame_extraction"] == ()
    assert STEP_DEPS["scene_detection"] == ()
    assert STEP_DEPS["visual_analysis"] == ("scene_detection",)
    assert STEP_DEPS["embeddings"] == ("scene_detection",)
    assert STEP_DEPS["llm_description"] == ("scene_detection",)
    assert STEP_ORDER == (
        "frame_extraction",
        "scene_detection",
        "visual_analysis",
        "embeddings",
        "llm_description",
    )


# ── Public API: successful run unchanged ──────────────────────────────────────


def test_full_run_all_steps_execute_in_order(tmp_path):
    p, cfg, calls = _pipeline_with_stubbed_steps(tmp_path, outcomes={})
    # scene_detection produces the keyframes dir + metadata so downstream
    # input gates pass.
    (cfg.paths.metadata_dir / "keyframes_metadata.json").write_text("[]")
    (cfg.paths.frames_dir / "scenes" / "keyframes_content" / "k.jpg").touch()

    res = p.run_steps("video.mp4", steps=list(STEP_ORDER))

    assert calls == list(STEP_ORDER)
    assert res.ok
    assert [r.state for r in res.runs] == ["done"] * 5


def test_progress_callback_reports_start_and_finish(tmp_path):
    p, cfg, _ = _pipeline_with_stubbed_steps(tmp_path, outcomes={})
    (cfg.paths.metadata_dir / "keyframes_metadata.json").write_text("[]")
    (cfg.paths.frames_dir / "scenes" / "keyframes_content" / "k.jpg").touch()

    events: list[tuple[str, str, str | None]] = []
    p.run_steps(
        "v.mp4",
        steps=list(STEP_ORDER),
        progress_cb=lambda n, ph, run: events.append(
            (n, ph, run.state if run else None)
        ),
    )
    # Each step: one start (run None) then one finish (run with state).
    fe = [e for e in events if e[0] == "frame_extraction"]
    assert fe == [
        ("frame_extraction", "start", None),
        ("frame_extraction", "finish", "done"),
    ]


# ── Dependency gating: the core defect fix ────────────────────────────────────


def test_scene_detection_failure_blocks_downstream_no_stale_output(tmp_path):
    """The historical defect: scene_detection fails but embeddings/llm
    still ran on a STALE keyframes_metadata.json. Gating must block them
    so NO downstream step executes (calls list proves it)."""
    p, cfg, calls = _pipeline_with_stubbed_steps(
        tmp_path, outcomes={"scene_detection": "fail"}
    )
    # Simulate a stale metadata file from a PRIOR run still on disk —
    # the old code would let embeddings/llm run on this.
    (cfg.paths.metadata_dir / "keyframes_metadata.json").write_text(
        '[{"stale": true}]'
    )

    res = p.run_steps("v.mp4", steps=list(STEP_ORDER))

    states = {r.name: r.state for r in res.runs}
    assert states["scene_detection"] == "error"
    assert states["visual_analysis"] == "blocked"
    assert states["embeddings"] == "blocked"
    assert states["llm_description"] == "blocked"
    # Proof no stale mixed output: the step impls were never invoked.
    assert "embeddings" not in calls
    assert "llm_description" not in calls
    assert "visual_analysis" not in calls
    assert not res.ok


def test_blocked_reason_is_explicit(tmp_path):
    p, _, _ = _pipeline_with_stubbed_steps(
        tmp_path, outcomes={"scene_detection": "fail"}
    )
    res = p.run_steps(
        "v.mp4", steps=["scene_detection", "embeddings"]
    )
    emb = next(r for r in res.runs if r.name == "embeddings")
    assert emb.state == "blocked"
    assert "scene_detection" in (emb.error or "")


def test_legitimate_subset_run_with_prior_artefacts_still_works(tmp_path):
    """A user runs ONLY embeddings, relying on keyframes_metadata.json
    from a prior successful run. Gating must NOT block this — inputs
    exist on disk and no in-run prerequisite contradicts them."""
    p, cfg, calls = _pipeline_with_stubbed_steps(tmp_path, outcomes={})
    (cfg.paths.metadata_dir / "keyframes_metadata.json").write_text("[]")

    res = p.run_steps("v.mp4", steps=["embeddings"])

    assert [r.state for r in res.runs] == ["done"]
    assert calls == ["embeddings"]
    assert res.ok


def test_subset_run_missing_inputs_is_blocked_not_crashed(tmp_path):
    """Run embeddings with NO metadata on disk and scene_detection not
    in this run: blocked with a clear reason, step never invoked."""
    p, _, calls = _pipeline_with_stubbed_steps(tmp_path, outcomes={})
    res = p.run_steps("v.mp4", steps=["embeddings"])
    r = res.runs[0]
    assert r.state == "blocked"
    assert "missing" in (r.error or "")
    assert calls == []


def test_visual_analysis_blocked_when_no_keyframe_files(tmp_path):
    p, _, calls = _pipeline_with_stubbed_steps(tmp_path, outcomes={})
    # No .jpg keyframes on disk, scene_detection not run.
    res = p.run_steps("v.mp4", steps=["visual_analysis"])
    assert res.runs[0].state == "blocked"
    assert "visual_analysis" not in calls


# ── Cancellation (pipeline level) ─────────────────────────────────────────────


def test_run_steps_raises_on_cancel_between_steps(tmp_path):
    p, cfg, calls = _pipeline_with_stubbed_steps(tmp_path, outcomes={})
    (cfg.paths.metadata_dir / "keyframes_metadata.json").write_text("[]")
    (cfg.paths.frames_dir / "scenes" / "keyframes_content" / "k.jpg").touch()

    state = {"n": 0}

    def cancel_check():
        state["n"] += 1
        return state["n"] > 2  # cancel after a couple of step boundaries

    with pytest.raises(StepCancelled):
        p.run_steps(
            "v.mp4", steps=list(STEP_ORDER), cancel_check=cancel_check
        )
    # Stopped early — not every step ran.
    assert len(calls) < len(STEP_ORDER)


# ── Job registry: lifecycle / retention / concurrency / thread-safety ─────────


@pytest.fixture()
def jobs_mod():
    import api.jobs as jobs

    # Fresh registry per test (mirror conftest's reset idiom).
    jobs._registry.reset()
    return jobs


class _StubPipeline:
    """Drop-in for CatalogPipeline used by the job runner tests.

    Its run_steps is controlled by class attributes so each test shapes
    success / failure / blocking / timing without any model.
    """

    behavior = "ok"  # ok | fail | block | hang

    def __init__(self, cfg):
        self.cfg = cfg

    def run_steps(self, video_path, steps, progress_cb=None, cancel_check=None):
        from cinemateca.pipeline import StepCancelled, StepResults, StepRun

        res = StepResults(video_path=str(video_path))
        for name in steps:
            if cancel_check is not None and cancel_check():
                raise StepCancelled()
            if self.behavior == "hang":
                # Cooperative: spin until cancelled.
                for _ in range(2000):
                    if cancel_check and cancel_check():
                        raise StepCancelled()
                    time.sleep(0.005)
            if progress_cb:
                progress_cb(name, "start", None)
            if self.behavior == "fail" and name == "scene_detection":
                run = StepRun(name=name, state="error", error="boom")
            elif self.behavior == "block" and name in (
                "embeddings",
                "llm_description",
                "visual_analysis",
            ):
                run = StepRun(name=name, state="blocked", error="prereq")
            else:
                run = StepRun(name=name, state="done", duration_s=0.5)
            res.runs.append(run)
            if progress_cb:
                progress_cb(name, "finish", run)
        return res


def _patch_pipeline(jobs_mod, monkeypatch, behavior="ok"):
    import cinemateca.pipeline as pl

    _StubPipeline.behavior = behavior
    monkeypatch.setattr(pl, "CatalogPipeline", _StubPipeline)


def _wait_terminal(job, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if job.is_terminal:
            return
        time.sleep(0.01)
    raise AssertionError(f"job did not reach terminal state: {job.status}")


def test_job_lifecycle_created_running_done(jobs_mod, monkeypatch):
    _patch_pipeline(jobs_mod, monkeypatch, "ok")
    jid = jobs_mod.start_job("v.mp4", {"frame_extraction"}, object())
    job = jobs_mod.get_job(jid)
    _wait_terminal(job)
    assert job.status == jobs_mod.STATUS_DONE
    assert job.progress == 1.0


def test_job_lifecycle_error_on_failed_step(jobs_mod, monkeypatch):
    _patch_pipeline(jobs_mod, monkeypatch, "fail")
    jid = jobs_mod.start_job(
        "v.mp4", {"scene_detection", "embeddings"}, object()
    )
    job = jobs_mod.get_job(jid)
    _wait_terminal(job)
    assert job.status == jobs_mod.STATUS_ERROR
    assert job.error_msg


def test_blocked_step_makes_job_error_no_silent_success(jobs_mod, monkeypatch):
    _patch_pipeline(jobs_mod, monkeypatch, "block")
    jid = jobs_mod.start_job(
        "v.mp4", {"scene_detection", "embeddings"}, object()
    )
    job = jobs_mod.get_job(jid)
    _wait_terminal(job)
    assert job.status == jobs_mod.STATUS_ERROR
    blocked = [s for s in job.steps if s.state == "blocked"]
    assert blocked, [s.state for s in job.steps]


def test_concurrency_single_global_active_job_rejected(jobs_mod, monkeypatch):
    """Second start while one runs is rejected (single-user policy)."""
    _patch_pipeline(jobs_mod, monkeypatch, "hang")
    jid = jobs_mod.start_job("a.mp4", {"frame_extraction"}, object())
    job = jobs_mod.get_job(jid)
    # Wait until it is actually running.
    t0 = time.time()
    while job.status != jobs_mod.STATUS_RUNNING and time.time() - t0 < 2:
        time.sleep(0.01)

    with pytest.raises(jobs_mod.ConcurrencyRejected) as ei:
        jobs_mod.start_job("b.mp4", {"frame_extraction"}, object())
    assert ei.value.active.id == jid

    # Cancel to release the worker.
    jobs_mod.cancel_job(jid)
    _wait_terminal(job)


def test_cancellation_mid_run_yields_cancelled_terminal(jobs_mod, monkeypatch):
    _patch_pipeline(jobs_mod, monkeypatch, "hang")
    jid = jobs_mod.start_job("v.mp4", {"frame_extraction"}, object())
    job = jobs_mod.get_job(jid)
    t0 = time.time()
    while job.status != jobs_mod.STATUS_RUNNING and time.time() - t0 < 2:
        time.sleep(0.01)

    assert jobs_mod.cancel_job(jid) is True
    _wait_terminal(job)
    assert job.status == jobs_mod.STATUS_CANCELLED
    # A terminal "cancelled" signal must be queued for the SSE stream.
    drained = []
    while not job.events.empty():
        drained.append(job.events.get_nowait())
    assert "cancelled" in drained, drained


def test_cancel_unknown_or_finished_job_returns_false(jobs_mod, monkeypatch):
    _patch_pipeline(jobs_mod, monkeypatch, "ok")
    assert jobs_mod.cancel_job("nope") is False
    jid = jobs_mod.start_job("v.mp4", {"frame_extraction"}, object())
    job = jobs_mod.get_job(jid)
    _wait_terminal(job)
    assert jobs_mod.cancel_job(jid) is False  # already terminal


def test_retention_evicts_oldest_terminal_jobs(jobs_mod, monkeypatch):
    _patch_pipeline(jobs_mod, monkeypatch, "ok")
    monkeypatch.setattr(jobs_mod._registry, "_max_terminal", 3)

    ids = []
    for _ in range(6):
        jid = jobs_mod.start_job("v.mp4", {"frame_extraction"}, object())
        ids.append(jid)
        _wait_terminal(jobs_mod.get_job(jid))

    remaining = {j.id for j in jobs_mod._registry.all()}
    # Only the last 3 terminal jobs are retained.
    assert len(remaining) == 3
    assert set(ids[-3:]) == remaining
    assert not (set(ids[:3]) & remaining)


def test_registry_thread_safe_concurrent_access(jobs_mod, monkeypatch):
    """Concurrent get/active reads while jobs are inserted must not
    raise / corrupt the registry (basic sanity, not a stress test)."""
    _patch_pipeline(jobs_mod, monkeypatch, "ok")
    errors: list[Exception] = []
    stop = threading.Event()

    def reader():
        try:
            while not stop.is_set():
                jobs_mod.active_jobs()
                jobs_mod._registry.all()
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    try:
        for _ in range(5):
            jid = jobs_mod.start_job("v.mp4", {"frame_extraction"}, object())
            _wait_terminal(jobs_mod.get_job(jid))
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=2)
    assert not errors, errors


# ── reset()/add() test-isolation hooks (Phase 7) ──────────────────────────────


def test_registry_reset_clears_jobs_preserving_container(jobs_mod):
    """``reset()`` empties the registry without swapping the dict object.

    Hermeticity contract for conftest/test_sse: after ``reset()`` no
    prior ``JobState`` is visible, and the ``_jobs`` container identity
    is preserved (``.clear()``, not reassignment) so any code holding a
    reference to it observes the empty state too.
    """
    reg = jobs_mod._registry
    job = jobs_mod.JobState(id="resetme", video_path="v.mp4")
    reg.add(job)
    assert reg.get("resetme") is job

    container_before = reg._jobs
    reg.reset()

    assert reg.get("resetme") is None
    assert reg.all() == []
    assert reg._jobs is container_before  # cleared in place, not rebound


def test_registry_add_inserts_under_lock(jobs_mod):
    """``add()`` registers a pre-built job retrievable via the public API."""
    reg = jobs_mod._registry
    reg.reset()
    job = jobs_mod.JobState(id="addme", video_path="v.mp4")
    job.steps = [
        jobs_mod.StepInfo(name=n, label=lbl) for n, lbl in jobs_mod.STEP_DEFS
    ]
    reg.add(job)

    assert jobs_mod.get_job("addme") is job
    assert job in reg.all()
