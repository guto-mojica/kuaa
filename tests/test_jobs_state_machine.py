"""A9: job lifecycle is an explicit, guarded state machine."""
from __future__ import annotations

import pytest

from api.jobs import JobState, JobStatus
from cinemateca.errors import PipelineError


def test_legal_transition_created_to_running() -> None:
    job = JobState(id="t1", video_path="/x.mp4", status=JobStatus.CREATED)
    job.transition_to(JobStatus.RUNNING)
    assert job.status is JobStatus.RUNNING


def test_running_to_done_is_legal() -> None:
    job = JobState(id="t2", video_path="/x.mp4", status=JobStatus.RUNNING)
    job.transition_to(JobStatus.DONE)
    assert job.status is JobStatus.DONE


def test_terminal_state_is_frozen() -> None:
    job = JobState(id="t3", video_path="/x.mp4", status=JobStatus.DONE)
    with pytest.raises(PipelineError):
        job.transition_to(JobStatus.RUNNING)


def test_status_string_aliases_preserved() -> None:
    # Back-compat: the old string constants still compare equal (str enum).
    assert JobStatus.RUNNING == "running"
    assert JobStatus.DONE == "done"


def test_created_to_cancelled_is_legal() -> None:
    """Cancel before runner picks up the job must be legal."""
    job = JobState(id="t4", video_path="/x.mp4", status=JobStatus.CREATED)
    job.transition_to(JobStatus.CANCELLED)
    assert job.status is JobStatus.CANCELLED


def test_running_to_error_is_legal() -> None:
    job = JobState(id="t5", video_path="/x.mp4", status=JobStatus.RUNNING)
    job.transition_to(JobStatus.ERROR)
    assert job.status is JobStatus.ERROR


def test_running_to_cancelled_is_legal() -> None:
    job = JobState(id="t6", video_path="/x.mp4", status=JobStatus.RUNNING)
    job.transition_to(JobStatus.CANCELLED)
    assert job.status is JobStatus.CANCELLED


def test_error_state_is_frozen() -> None:
    job = JobState(id="t7", video_path="/x.mp4", status=JobStatus.ERROR)
    with pytest.raises(PipelineError):
        job.transition_to(JobStatus.RUNNING)


def test_cancelled_state_is_frozen() -> None:
    job = JobState(id="t8", video_path="/x.mp4", status=JobStatus.CANCELLED)
    with pytest.raises(PipelineError):
        job.transition_to(JobStatus.DONE)


def test_illegal_transition_message_includes_states() -> None:
    job = JobState(id="t9", video_path="/x.mp4", status=JobStatus.DONE)
    with pytest.raises(PipelineError, match="done.*running"):
        job.transition_to(JobStatus.RUNNING)


def test_status_enum_aliases_constants() -> None:
    """Module-level STATUS_* constants are enum aliases (same object or equal)."""
    from api.jobs import (
        STATUS_CANCELLED,
        STATUS_CREATED,
        STATUS_DONE,
        STATUS_ERROR,
        STATUS_RUNNING,
    )

    assert STATUS_CREATED == "created"
    assert STATUS_RUNNING == "running"
    assert STATUS_DONE == "done"
    assert STATUS_ERROR == "error"
    assert STATUS_CANCELLED == "cancelled"
    # They must be the enum members themselves (not bare strings).
    assert isinstance(STATUS_CREATED, JobStatus)
    assert isinstance(STATUS_RUNNING, JobStatus)
    assert isinstance(STATUS_DONE, JobStatus)
    assert isinstance(STATUS_ERROR, JobStatus)
    assert isinstance(STATUS_CANCELLED, JobStatus)
