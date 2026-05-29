"""Cinemateca error taxonomy (F2).

One base ``CinematecaError`` carrying a stable ``.code`` and a flat
subtree the HTTP layer (WS-2 A4) maps to status codes via
:func:`http_status_for`. Existing scattered exceptions migrate to inherit
these (keeping their names as aliases) in a follow-up step of this task.
"""
from __future__ import annotations


class CinematecaError(Exception):
    """Base for every domain error. Carries a stable ``code``.

    ``code`` defaults to the class's ``default_code`` (a dotted machine
    string) but can be overridden per raise site for finer-grained
    client handling.
    """

    default_code: str = "cinemateca.error"

    def __init__(self, *args: object, code: str | None = None) -> None:
        super().__init__(*args)
        self.code: str = code or self.default_code


class ConfigError(CinematecaError):
    """Configuration is missing, malformed, or fails schema validation."""

    default_code = "config.invalid"


class ModelError(CinematecaError):
    """A model backend failed to load or run."""

    default_code = "model.failure"


class PipelineError(CinematecaError):
    """A pipeline step failed in a way the caller must surface."""

    default_code = "pipeline.failure"


class RetrievalError(CinematecaError):
    """Search/retrieval could not be completed."""

    default_code = "retrieval.failure"


class IndexMissing(RetrievalError):
    """A required search index is absent on disk (empty-state signal)."""

    default_code = "retrieval.index_missing"


class UserInputError(CinematecaError):
    """Client supplied invalid input (bad upload, bad slug, bad query)."""

    default_code = "input.invalid"


class ArtefactError(CinematecaError):
    """A generated artefact is missing or unreadable."""

    default_code = "artefact.invalid"


# Single source of truth for HTTP status. WS-2 A4's exception handler
# imports this; nothing else hard-codes a status for a domain error.
_STATUS_TABLE: tuple[tuple[type[CinematecaError], int], ...] = (
    (UserInputError, 400),
    (IndexMissing, 404),
    (ArtefactError, 500),
    (ConfigError, 500),
    (ModelError, 500),
    (PipelineError, 500),
    (RetrievalError, 500),
)


def http_status_for(exc: BaseException) -> int:
    """Return the HTTP status for ``exc`` (most specific subclass wins).

    Non-:class:`CinematecaError` exceptions map to 500.
    """
    best: int = 500
    best_depth = -1
    for cls, status in _STATUS_TABLE:
        if isinstance(exc, cls):
            depth = len(cls.__mro__)
            if depth > best_depth:
                best, best_depth = status, depth
    return best


__all__ = [
    "CinematecaError",
    "ConfigError",
    "ModelError",
    "PipelineError",
    "RetrievalError",
    "IndexMissing",
    "UserInputError",
    "ArtefactError",
    "http_status_for",
]
