"""E0: EvalError lives in kuaa.errors and is re-exported correctly.

Three invariants:
1. Canonical home is ``kuaa.errors`` (issubclass chain + code).
2. ``kuaa.eval.__init__`` re-exports it as a true alias (not a copy).
3. ``kuaa.eval.retrieval`` still exposes it for back-compat (not a copy).
"""

from __future__ import annotations


def test_eval_error_is_kuaa_error() -> None:
    from kuaa.errors import KuaaError, EvalError, RetrievalError

    assert issubclass(EvalError, KuaaError)
    assert issubclass(EvalError, RetrievalError)
    assert EvalError("x").code == "eval.failure"


def test_eval_error_reexported_from_eval_package() -> None:
    from kuaa.errors import EvalError as E2
    from kuaa.eval import EvalError as E1

    assert E1 is E2


def test_eval_retrieval_still_exposes_eval_error() -> None:
    from kuaa.errors import EvalError as E2
    from kuaa.eval.retrieval import EvalError as E3

    assert E3 is E2
