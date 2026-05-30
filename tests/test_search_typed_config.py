"""C2 — WS-1 public entry points are typed with Settings, not Any.

The modules under test use ``from __future__ import annotations``, which
causes ``inspect.signature`` to return annotation *strings* rather than
resolved types.  So ``inspect.signature(fn).parameters["cfg"].annotation``
would return the string ``"Settings"``, not the class — making a bare
``_ann(fn, "cfg") is Settings`` assertion vacuously fail (string != type).

Fix: use ``typing.get_type_hints(fn)`` which evaluates deferred string
annotations in the function's module namespace, returning real type objects.
This makes the tests *genuinely* verify that ``cfg`` is annotated as
``Settings`` (or ``Settings | None``), not vacuously pass because strings
are always != types.

Additional gotcha: ``from cinemateca.search import aggregate`` imports the
*function* ``aggregate`` via the package ``__init__.py``, not the submodule.
We use ``importlib.import_module`` to obtain the actual module objects.

Non-vacuity guarantee: reverting any one ``cfg: Settings`` back to
``cfg: Any`` causes ``get_type_hints`` to return ``typing.Any`` for that
parameter, and the corresponding test fails.
"""

from __future__ import annotations

import importlib
import typing

from cinemateca.config import Settings

# Import the submodules directly — ``from cinemateca.search import aggregate``
# would give the *function*, not the module, because __init__.py re-exports it.
_agg_mod = importlib.import_module("cinemateca.search.aggregate")
_dispatch_mod = importlib.import_module("cinemateca.search._dispatch")


def _ann(fn: object, param: str) -> object:
    """Return the *resolved* annotation for ``param`` in ``fn``.

    Uses ``typing.get_type_hints`` so that ``from __future__ import
    annotations`` deferred strings are evaluated in the function's module
    namespace.  Returns ``None`` if the parameter is not annotated.
    """
    hints = typing.get_type_hints(fn)  # type: ignore[arg-type]
    return hints.get(param)


def test_aggregate_search_cfg_is_settings() -> None:
    """aggregate_search's cfg parameter must be annotated Settings (not Any)."""
    ann = _ann(_agg_mod.aggregate_search, "cfg")
    assert ann is Settings, f"expected Settings, got {ann!r}"


def test_aggregate_public_verb_cfg_is_settings() -> None:
    """aggregate's cfg keyword parameter must be annotated Settings (not Any)."""
    ann = _ann(_agg_mod.aggregate, "cfg")
    assert ann is Settings, f"expected Settings, got {ann!r}"


def test_has_indexed_films_cfg_is_settings() -> None:
    """has_indexed_films' cfg parameter must be annotated Settings (not Any)."""
    ann = _ann(_agg_mod.has_indexed_films, "cfg")
    assert ann is Settings, f"expected Settings, got {ann!r}"


def test_aggregate_hits_to_template_dicts_cfg_is_settings() -> None:
    """aggregate_hits_to_template_dicts' cfg parameter must be annotated Settings."""
    ann = _ann(_agg_mod.aggregate_hits_to_template_dicts, "cfg")
    assert ann is Settings, f"expected Settings, got {ann!r}"


def test_find_cfg_is_settings_optional() -> None:
    """find's cfg parameter must be annotated Settings | None (not Any)."""
    ann = _ann(_dispatch_mod.find, "cfg")
    # get_type_hints resolves "Settings | None" as Optional[Settings]
    assert ann == (Settings | None), f"expected Optional[Settings], got {ann!r}"


def test_no_bare_Any_in_aggregate_public_signatures() -> None:
    """No public aggregate-module entry point should have a bare cfg: Any."""
    for name in (
        "aggregate_search",
        "aggregate",
        "has_indexed_films",
        "aggregate_hits_to_template_dicts",
    ):
        fn = getattr(_agg_mod, name)
        hints = typing.get_type_hints(fn)  # type: ignore[arg-type]
        cfg_ann = hints.get("cfg")
        assert cfg_ann is not typing.Any, f"{name}.cfg is still annotated as Any — must be Settings"
        assert cfg_ann is not None, f"{name} has no 'cfg' annotation at all"
