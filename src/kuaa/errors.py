"""KUAA error taxonomy (F2).

One base ``KuaaError`` carrying a stable ``.code`` and a flat
subtree the HTTP layer (WS-2 A4) maps to status codes via
:func:`http_status_for`. Existing scattered exceptions migrate to inherit
these (keeping their names as aliases) in a follow-up step of this task.
"""

from __future__ import annotations

import re


class KuaaError(Exception):
    """Base for every domain error. Carries a stable ``code``.

    ``code`` defaults to the class's ``default_code`` (a dotted machine
    string) but can be overridden per raise site for finer-grained
    client handling.
    """

    default_code: str = "kuaa.error"

    def __init__(self, *args: object, code: str | None = None) -> None:
        super().__init__(*args)
        self.code: str = code or self.default_code


class ConfigError(KuaaError):
    """Configuration is missing, malformed, or fails schema validation."""

    default_code = "config.invalid"


class ModelError(KuaaError):
    """A model backend failed to load or run."""

    default_code = "model.failure"


class PipelineError(KuaaError):
    """A pipeline step failed in a way the caller must surface."""

    default_code = "pipeline.failure"


class RetrievalError(KuaaError):
    """Search/retrieval could not be completed."""

    default_code = "retrieval.failure"


class IndexMissing(RetrievalError):
    """A required search index is absent on disk (empty-state signal)."""

    default_code = "retrieval.index_missing"


class EvalError(RetrievalError):
    """Raised for clear user-facing evaluation failures."""

    default_code = "eval.failure"


class UserInputError(KuaaError):
    """Client supplied invalid input (bad upload, bad slug, bad query)."""

    default_code = "input.invalid"


class ArtefactError(KuaaError):
    """A generated artefact is missing or unreadable."""

    default_code = "artefact.invalid"


# Single source of truth for HTTP status. WS-2 A4's exception handler
# imports this; nothing else hard-codes a status for a domain error.
_STATUS_TABLE: tuple[tuple[type[KuaaError], int], ...] = (
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

    Non-:class:`KuaaError` exceptions map to 500.
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
    "KuaaError",
    "ConfigError",
    "ModelError",
    "PipelineError",
    "RetrievalError",
    "IndexMissing",
    "EvalError",
    "UserInputError",
    "ArtefactError",
    "http_status_for",
    "is_error_response",
    "is_degenerate_response",
    "is_unusable_response",
]


# Backend failures are captured as strings and flow into the same fields as
# real answers, so an exception can end up indexed as scene content. This
# actually happened: every scene of ``the-great-train-robbery-1903`` carries
# the tag ``error:-passed-cpu-tensor-to-mps-op``, an MPS device error that
# became a searchable BM25 term. The pre-existing ``startswith("ERROR")``
# check missed it because the backend emitted lowercase ``Error:``.
_ERROR_RE = re.compile(
    r"^\s*(error|erro|exception|traceback|runtimeerror|valueerror|notimplementederror)\b"
    r"|^\s*<[a-z_]*error",
    flags=re.IGNORECASE,
)


def is_error_response(text: str) -> bool:
    """True when a model response is a captured failure, not an answer.

    Guards every free-text field that reaches the tag vocabulary. Matching
    is case-insensitive and anchored at the start, so a caption that merely
    mentions the word "error" mid-sentence is still indexed.
    """
    return bool(text) and bool(_ERROR_RE.search(text))


# A VLM whose device is failing does not raise — it emits fluent-looking
# garbage. Under MPS memory exhaustion the decoder's logits go to noise,
# argmax pins on one token, EOS is never reached, and every prompt runs to
# the model's token ceiling. Observed 2026-08-14 on ``chronopolis_1982``:
# scene 1 described perfectly, then 19 consecutive scenes came back as
# ``"s. s. s. s. ..."`` — several hundred tokens, one distinct word.
#
# That text passes :func:`is_error_response` (it does not *start* with an
# error word) and every other guard, so it would have been written to
# scene_descriptions.json, kebab-cased into the tag vocabulary, and indexed
# as BM25 scene content. Only a checkpoint interval (25) larger than the
# scene count reached (20) kept it off disk.
#
# Thresholds are deliberately conservative and were validated against real
# artefacts: zero false positives across the 1320 raw prompt answers and 339
# tags of ``band_of_outsiders`` (a healthy run), and all 19 degenerate scenes
# of the failed ``chronopolis_1982`` run caught — scene 1, which was healthy,
# correctly not flagged.
_DEGENERATE_MIN_TOKENS = 8
_DEGENERATE_MAX_UNIQUE = 3
_DEGENERATE_RATIO_MIN_TOKENS = 20
_DEGENERATE_MAX_UNIQUE_RATIO = 0.15

_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


def is_degenerate_response(text: str) -> bool:
    """True when a model response is a repetition loop, not an answer.

    Two arms, both requiring enough length that a healthy terse answer
    ("outdoor", "2 people talking") can never trip them:

    * **short loop** — at least 8 tokens carrying 3 or fewer distinct words.
      Catches the single-token pin that device failure produces.
    * **long loop** — at least 20 tokens whose distinct-word ratio is under
      0.15. Catches a decoder cycling through a small set of tokens.

    Case- and punctuation-insensitive: ``"s. s. s."`` and ``"s s s"`` are
    the same failure and must both be caught.
    """
    if not text:
        return False
    tokens = [t.lower() for t in _WORD_RE.findall(text)]
    if len(tokens) < _DEGENERATE_MIN_TOKENS:
        return False
    unique = len(set(tokens))
    if unique <= _DEGENERATE_MAX_UNIQUE:
        return True
    return (
        len(tokens) >= _DEGENERATE_RATIO_MIN_TOKENS
        and unique / len(tokens) < _DEGENERATE_MAX_UNIQUE_RATIO
    )


def is_unusable_response(text: str) -> bool:
    """True when a model response must never reach the index.

    The single gate for free text on its way into descriptions, tags, or the
    BM25 corpus: a captured backend failure (:func:`is_error_response`) or a
    repetition loop (:func:`is_degenerate_response`).
    """
    return is_error_response(text) or is_degenerate_response(text)
