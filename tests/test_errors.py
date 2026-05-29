"""Error taxonomy contract tests (F2)."""

from __future__ import annotations

import pytest

from cinemateca.errors import (
    ArtefactError,
    CinematecaError,
    ConfigError,
    IndexMissing,
    ModelError,
    PipelineError,
    RetrievalError,
    UserInputError,
    http_status_for,
)

ALL_SUBCLASSES = [
    ConfigError,
    ModelError,
    PipelineError,
    RetrievalError,
    IndexMissing,
    UserInputError,
    ArtefactError,
]


@pytest.mark.parametrize("cls", ALL_SUBCLASSES)
def test_every_subclass_inherits_base_and_carries_code(cls):
    err = cls("boom")
    assert isinstance(err, CinematecaError)
    assert isinstance(err.code, str) and err.code  # non-empty stable code
    assert str(err) == "boom"


def test_index_missing_is_a_retrieval_error():
    assert issubclass(IndexMissing, RetrievalError)


def test_code_is_overridable_per_instance():
    assert ConfigError("x", code="config.bad_key").code == "config.bad_key"


@pytest.mark.parametrize(
    "cls,status",
    [
        (UserInputError, 400),
        (IndexMissing, 404),
        (ConfigError, 500),
        (ModelError, 500),
        (PipelineError, 500),
        (RetrievalError, 500),
        (ArtefactError, 500),
    ],
)
def test_http_status_mapping(cls, status):
    assert http_status_for(cls("e")) == status


def test_http_status_for_unknown_defaults_500():
    assert http_status_for(ValueError("nope")) == 500
