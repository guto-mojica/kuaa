"""E2E loading-skeleton runtime tests (U2) — visibility in flight vs idle.

``tests/test_loading_skeletons.py`` pins the markup + CSS (the ``.skeleton``
primitive, the ``prefers-reduced-motion`` neutraliser, the ``hx-indicator``
gating rules, and each pane's placeholder element). What it cannot see is
whether the skeleton actually SHOWS while a request is in flight and HIDES when
idle — htmx toggles ``.htmx-request`` on the indicator target at runtime, so
only a browser can verify the reveal. That is this module's job.

To make the in-flight window deterministic (and to avoid loading the heavy
CLIP/SigLIP weights a real search would pull), we intercept the endpoint with a
controlled delay via ``page.route`` and watch the skeleton with an in-page
``MutationObserver`` that records whether it was EVER visible between the
trigger and the response. A peak-visibility observer (rather than coarse
Python-side polling) is the robust check here: htmx's ``hx-sync`` aborts and
re-issues the request, so ``.htmx-request`` flickers on the indicator faster
than a poll loop can sample — but the observer catches every transition. We
then confirm the skeleton is hidden once the response settles.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from .conftest import wait_for_alpine

pytestmark = pytest.mark.e2e

# Minimal valid swap body: htmx fulfils the stalled request with this so the
# swap completes cleanly and the indicator clears. We only need a 200 with a
# body htmx will accept for the target.
_EMPTY_RESULTS = "<!-- e2e: empty results -->"


def _install_peak_observer(page: Any, skeleton_id: str) -> None:
    """Arm a MutationObserver that flips ``window.__peakVisible`` the first
    time *skeleton_id* is rendered (``display != none``).

    Observing the indicator's ``class`` / ``style`` mutations captures the
    brief in-flight window even when htmx's request class flickers (abort +
    re-issue under ``hx-sync``) faster than a poll could sample.
    """
    page.evaluate(
        """(id) => {
            const el = document.getElementById(id);
            window.__peakVisible = false;
            const check = () => {
                if (el && getComputedStyle(el).display !== 'none') {
                    window.__peakVisible = true;
                }
            };
            window.__skelObs = new MutationObserver(check);
            window.__skelObs.observe(el, { attributes: true, attributeFilter: ['class', 'style'] });
            check();
        }""",
        skeleton_id,
    )


def _peak_visible(page: Any) -> bool:
    return bool(page.evaluate("() => window.__peakVisible === true"))


def _is_hidden(page: Any, skeleton_id: str) -> bool:
    return (
        page.evaluate(
            "(id) => { const el = document.getElementById(id);"
            " return el ? getComputedStyle(el).display === 'none' : true; }",
            skeleton_id,
        )
        is True
    )


def _stall_handler(route: Any) -> None:
    """Hold the request ~0.8 s, then return an empty swap body (200)."""
    time.sleep(0.8)
    route.fulfill(status=200, content_type="text/html; charset=utf-8", body=_EMPTY_RESULTS)


def test_search_skeleton_visible_in_flight_hidden_when_idle(page: Any) -> None:
    """``#search-skeleton`` is hidden at rest, shown during a search, hidden after.

    A ``page.route`` shim stalls ``/api/search`` ~0.8 s; the peak-visibility
    observer confirms the skeleton was revealed while the request was in flight,
    and the post-settle check confirms it hides again.
    """
    page.goto("/search", wait_until="domcontentloaded")
    wait_for_alpine(page)

    # At rest the skeleton is not shown.
    assert _is_hidden(page, "search-skeleton")

    page.route("**/api/search?*", _stall_handler)
    _install_peak_observer(page, "search-skeleton")

    # Submit the search via the form's submit button (drives the real
    # hx-indicator="#search-skeleton" wiring).
    page.locator("#search-input").fill("a river at dusk")
    page.locator("#search-text-form button[type=submit]").click()

    # The skeleton must become visible at some point during the in-flight
    # request (poll the observer flag, which latches on the first reveal).
    page.wait_for_function("() => window.__peakVisible === true", timeout=5000)
    assert _peak_visible(page) is True

    # Once the response settles, the indicator clears and the skeleton hides.
    page.wait_for_function(
        """() => {
            const el = document.getElementById('search-skeleton');
            return el && getComputedStyle(el).display === 'none';
        }""",
        timeout=5000,
    )
    assert _is_hidden(page, "search-skeleton")

    page.unroute("**/api/search?*")


def test_scene_grid_skeleton_visible_in_flight(page: Any) -> None:
    """``#scenes-skeleton`` shows while an ``/api/scenes`` refresh is in flight.

    Driven on the Cenas tab against the real library. The ``#scenes-toolrow``
    carries ``hx-trigger="refresh"`` + ``hx-indicator="#scenes-skeleton"``; we
    stall the endpoint and fire that ``refresh`` event (the same path the
    appearance/sort popovers use), then assert the skeleton reveals via the
    peak observer and clears after the swap settles.
    """
    page.goto("/scenes", wait_until="domcontentloaded")
    page.wait_for_function("() => !!window.htmx", timeout=5000)
    assert _is_hidden(page, "scenes-skeleton")

    page.route("**/api/scenes?*", _stall_handler)
    _install_peak_observer(page, "scenes-skeleton")

    # Fire the toolrow's declarative refresh (uses hx-indicator="#scenes-skeleton").
    page.evaluate("() => window.htmx.trigger('#scenes-toolrow', 'refresh')")

    page.wait_for_function("() => window.__peakVisible === true", timeout=5000)
    assert _peak_visible(page) is True

    page.wait_for_function(
        """() => {
            const el = document.getElementById('scenes-skeleton');
            return el && getComputedStyle(el).display === 'none';
        }""",
        timeout=5000,
    )
    assert _is_hidden(page, "scenes-skeleton")
    page.unroute("**/api/scenes?*")
