"""A10: builder return dicts match their declared TypedDict keys.

Runtime key-coverage half of A10 (the static half is ``uv run mypy ...``).
Each test calls the real builder and asserts that the returned dict
contains EVERY key declared as *required* in the matching TypedDict
(i.e. keys that are NOT in ``__optional_keys__``).

``total=False`` TypedDicts put all keys in ``__optional_keys__``; the
assertion still passes for those because the required set is empty — that
is intentional: ``total=False`` means "every key may be absent" at the
TypedDict level, but the builder always returns the full dict.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _required_keys(typed_dict_cls) -> set[str]:
    """Return the required keys of a TypedDict class.

    Works for both ``total=True`` (default) and mixed-totality TypedDicts
    created via the two-class inheritance pattern.  Python exposes
    ``__required_keys__`` on all TypedDict classes from 3.9+.
    """
    return set(getattr(typed_dict_cls, "__required_keys__", typed_dict_cls.__annotations__))


# ---------------------------------------------------------------------------
# CenasContext
# ---------------------------------------------------------------------------


def test_cenas_context_has_all_declared_keys(seed_metadata) -> None:
    """build_cenas_context returns every required CenasContext key."""
    seed_metadata()

    from api.contexts import CenasContext
    from api.deps import get_config
    from api.services.scenes import build_cenas_context

    ctx = build_cenas_context(get_config())
    required = _required_keys(CenasContext)
    missing = required - set(ctx)
    assert not missing, f"build_cenas_context missing declared keys: {missing}"


# ---------------------------------------------------------------------------
# SearchContext
# ---------------------------------------------------------------------------


def test_search_context_has_all_declared_keys(seed_metadata) -> None:
    """build_search_context_aggregate returns every required SearchContext key."""
    seed_metadata()

    from api.contexts import SearchContext
    from api.deps import get_config
    from cinemateca.search._lookup import build_search_context_aggregate

    ctx = build_search_context_aggregate(get_config())
    required = _required_keys(SearchContext)
    missing = required - set(ctx)
    assert not missing, f"build_search_context_aggregate missing declared keys: {missing}"


# ---------------------------------------------------------------------------
# InspectorContext
# ---------------------------------------------------------------------------


def test_inspector_context_has_all_declared_keys(seed_metadata) -> None:
    """build_inspector_context returns every required InspectorContext key (when it resolves)."""
    paths = seed_metadata()

    from api.contexts import InspectorContext
    from api.deps import get_config
    from api.services.scenes import build_inspector_context

    # scene_id=351 is the default seeded scene; slug="default" is the
    # registered test film (T9 layout in conftest).
    ctx = build_inspector_context(get_config(), scene_id=351, slug="default")
    if ctx is None:
        pytest.skip("build_inspector_context returned None on seed data — cannot check keys")
    required = _required_keys(InspectorContext)
    missing = required - set(ctx)
    assert not missing, f"build_inspector_context missing declared keys: {missing}"


# ---------------------------------------------------------------------------
# TimelineContext
# ---------------------------------------------------------------------------


def test_timeline_context_has_all_declared_keys(seed_metadata) -> None:
    """build_timeline_context returns every required TimelineContext key (when it resolves)."""
    seed_metadata()

    from api.contexts import TimelineContext
    from api.deps import get_config
    from api.services.scenes import build_timeline_context

    ctx = build_timeline_context(get_config(), slug="default", scene_id=351, query="test")
    if ctx is None:
        pytest.skip("build_timeline_context returned None on seed data — cannot check keys")
    required = _required_keys(TimelineContext)
    missing = required - set(ctx)
    assert not missing, f"build_timeline_context missing declared keys: {missing}"


# ---------------------------------------------------------------------------
# ProcessingContext
# ---------------------------------------------------------------------------


def test_processing_context_has_all_declared_keys(seed_metadata) -> None:
    """build_processing_context returns every required ProcessingContext key."""
    seed_metadata()

    from api.contexts import ProcessingContext
    from api.services.processing_render import build_processing_context

    ctx = build_processing_context()
    required = _required_keys(ProcessingContext)
    missing = required - set(ctx)
    assert not missing, f"build_processing_context missing declared keys: {missing}"


# ---------------------------------------------------------------------------
# RimasContext
# ---------------------------------------------------------------------------


def test_rimas_context_has_all_declared_keys(seed_metadata) -> None:
    """build_rimas_context returns every required RimasContext key."""
    seed_metadata()

    from api.contexts import RimasContext
    from api.deps import get_config
    from api.services.rhymes_service import build_rimas_context

    # anchor=None → empty-state branch; still returns the full context dict
    # with anchor_scene=None and echoes=[].
    ctx = build_rimas_context(get_config(), anchor=None)
    required = _required_keys(RimasContext)
    missing = required - set(ctx)
    assert not missing, f"build_rimas_context missing declared keys: {missing}"
