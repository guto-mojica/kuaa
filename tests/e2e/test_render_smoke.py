"""E2E render smoke — every main tab loads in a real browser, no JS errors.

Each of the five chrome surfaces (Buscar / Cenas / Anotar / Rimas + the About
modal) is opened in headless Chromium and asserted to:

  * navigate to the route without a non-2xx top response;
  * paint a key DOM landmark for that tab (proof the template rendered, not a
    blank error page);
  * raise NO uncaught page errors and NO ``console.error`` output (proof the
    vendored JS — htmx, Alpine, mojica.js, focus_trap.js — initialised cleanly
    against the real shell).

This is the browser-level complement to the hermetic ``test_web_routes`` /
``test_template_contexts`` route smokes: those prove the server emits the
markup; this proves a browser can actually run the page.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from .conftest import run_axe

pytestmark = pytest.mark.e2e


class _ConsoleGuard:
    """Collects ``console.error`` lines and uncaught page errors for a page."""

    def __init__(self) -> None:
        self.console_errors: list[str] = []
        self.page_errors: list[str] = []

    def attach(self, page: Any) -> None:
        page.on(
            "console",
            lambda msg: (
                self.console_errors.append(f"{msg.type}: {msg.text}")
                if msg.type == "error"
                else None
            ),
        )
        page.on("pageerror", lambda exc: self.page_errors.append(str(exc)))

    def assert_clean(self) -> None:
        assert not self.page_errors, f"uncaught page errors: {self.page_errors}"
        assert not self.console_errors, f"console.error output: {self.console_errors}"


@pytest.fixture()
def guarded_page(page: Any) -> Iterator[Any]:
    """A pytest-playwright ``page`` with a console/pageerror guard attached.

    The guard is wired BEFORE any navigation so it catches errors thrown during
    initial script execution. Tests call ``page.console_guard.assert_clean()``
    after the page settles.
    """
    guard = _ConsoleGuard()
    guard.attach(page)
    page.console_guard = guard  # type: ignore[attr-defined]
    yield page


# (route, a CSS selector that must exist once the tab has rendered)
_TABS = [
    ("/search", "#search-text-form"),
    ("/scenes", ".tab-panel"),
    ("/annotate", ".tab-panel"),
    ("/rimas", ".tab-panel"),
]


@pytest.mark.parametrize("route, selector", _TABS, ids=[t[0] for t in _TABS])
def test_tab_renders_without_js_errors(guarded_page: Any, route: str, selector: str) -> None:
    """Each main tab loads 200, paints its landmark, and logs no JS errors."""
    resp = guarded_page.goto(route, wait_until="domcontentloaded")
    assert resp is not None, f"no response for {route}"
    assert resp.ok, f"{route} returned HTTP {resp.status}"
    # The shared chrome landmark is always present …
    assert guarded_page.locator("header.ch-top").count() == 1, "chrome topbar missing"
    # … plus the per-tab key element.
    guarded_page.wait_for_selector(selector, state="attached", timeout=5000)
    # The polish-layer overlays are server-rendered into every full page.
    assert guarded_page.locator("#palette").count() == 1, "command palette scaffold missing"
    assert guarded_page.locator("#help").count() == 1, "help overlay scaffold missing"
    guarded_page.console_guard.assert_clean()


def test_buscar_renders_real_library_chrome(guarded_page: Any) -> None:
    """Buscar shows the live shell: search box, modality chips, knob popovers.

    Asserts the real-data path renders the interactive chrome (not just an
    empty placeholder) — the search input, the four modality chips, and at
    least one toolbar ``.popover`` (the focus-trap negative control surface).
    """
    guarded_page.goto("/search", wait_until="domcontentloaded")
    assert guarded_page.locator("#search-input").count() == 1
    # Four modality chips (text / image / audio / fusion).
    assert guarded_page.locator(".modes .chip").count() == 4
    # Toolbar popovers exist (the U3 negative-control surface).
    assert guarded_page.locator(".knob-popover .popover").count() >= 1
    guarded_page.console_guard.assert_clean()


def test_about_modal_renders_via_htmx(guarded_page: Any) -> None:
    """Clicking About swaps the modal into #modal-container and it renders.

    The About link (``a.about`` in the left pane) is an HTMX GET into
    ``#modal-container``; this drives the real swap and confirms the dialog
    paints with its title + the modal's focus-trap contract attributes.
    """
    guarded_page.goto("/search", wait_until="domcontentloaded")
    guarded_page.locator("a.about").click()
    about = guarded_page.locator("#about")
    about.wait_for(state="visible", timeout=5000)
    assert about.get_attribute("role") == "dialog"
    assert about.get_attribute("aria-modal") == "true"
    assert about.get_attribute("data-focus-trap") is not None
    assert guarded_page.locator("#about-title").is_visible()
    guarded_page.console_guard.assert_clean()


def test_run_axe_helper_executes_against_live_page(page: Any, axe_source: str) -> None:
    """The ``run_axe`` helper injects axe-core and returns a violations list.

    This is the harness self-test (NOT the zero-violations assertion — that is
    U5's gate). It proves the vendored axe.min.js loads, runs, and the bridge
    returns a JSON-serialisable list of violation objects with the expected
    shape, so U5 can build its audit on a known-good mechanism.
    """
    page.goto("/search", wait_until="domcontentloaded")
    violations = run_axe(page, axe_source)
    assert isinstance(violations, list)
    # Every entry (if any) is a well-formed axe violation object.
    for v in violations:
        assert "id" in v and "impact" in v and "nodes" in v
    # axe itself is reachable on the page (sanity that injection happened).
    assert page.evaluate("() => !!(window.axe && window.axe.version)") is True
