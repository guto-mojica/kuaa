"""E2E published a11y gate (U5) — ZERO serious/critical axe violations.

This is the enforcing half of the U5 accessibility audit. ``docs/ACCESSIBILITY.md``
is the human-readable conformance statement; this module is the executable gate
that keeps the statement true: it injects axe-core 4.10.2 (vendored, pinned, no
network) into a real headless-Chromium render of every shipped surface and
asserts axe reports no ``serious`` or ``critical`` violation on any of them.

Surfaces covered (the spec's "5 tabs + the modals"):
  * the four main tabs   — ``/search`` ``/scenes`` ``/annotate`` ``/rimas``;
  * the standalone About page ``/about`` (JS-off fallback render);
  * the About modal (HTMX-swapped over a tab);
  * the command palette (⌘K) and the keyboard-help overlay (?), each OPENED
    before auditing so the audit sees their live, visible DOM.

Why serious/critical only: those two impact tiers are the WCAG-blocking
failures (missing names, ARIA-contract breaks, sub-threshold contrast,
keyboard-inaccessible regions). ``minor``/``moderate`` findings are tracked as
polish in ACCESSIBILITY.md's "known limitations" but are not release-gating, so
pinning the gate at serious+critical keeps it meaningful and stable. The audit
that informed the fixes drove every surface to ZERO total violations; this gate
is set at the documented serious/critical bar so a future low-severity finding
(e.g. a new moderate heading-order nit) doesn't spuriously red the build while a
real regression still trips it immediately.

The before→after counts (the U5 fix landed all of these at 0) are recorded in
``docs/ACCESSIBILITY.md``. Re-run locally with ``just e2e`` (or
``uv run pytest -m e2e -q``).
"""

from __future__ import annotations

from typing import Any

import pytest

from .conftest import run_axe, wait_for_alpine

pytestmark = pytest.mark.e2e


# Impact tiers that fail the gate. axe impact is one of
# null/minor/moderate/serious/critical; we block the top two.
_BLOCKING = {"serious", "critical"}


def _blocking(violations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter an axe violations list to the serious/critical entries."""
    return [v for v in violations if v.get("impact") in _BLOCKING]


def _explain(violations: list[dict[str, Any]]) -> str:
    """Render a readable failure message: rule id, impact, and node targets.

    Keeps the assertion diagnosable — a regression names the rule and the exact
    element(s) so the fix is obvious without re-running axe by hand.
    """
    lines: list[str] = []
    for v in violations:
        targets = "; ".join(",".join(node.get("target", [])) for node in v.get("nodes", []))
        lines.append(f"  [{v.get('impact')}] {v.get('id')}: {v.get('help')}\n    → {targets}")
    return "\n".join(lines)


def _assert_axe_clean(page: Any, axe_source: str, *, surface: str) -> None:
    """Run axe on the current page state and assert zero serious/critical."""
    violations = run_axe(page, axe_source)
    blocking = _blocking(violations)
    assert (
        not blocking
    ), f"{surface}: {len(blocking)} serious/critical axe violation(s):\n" + _explain(blocking)


# ── Main tabs + standalone About page ──────────────────────────────────────

# (route, a selector to wait for so the audit runs against a settled DOM)
_PAGES = [
    ("/search", "#search-text-form"),
    ("/scenes", ".tab-panel"),
    ("/annotate", ".tab-panel"),
    ("/rimas", ".tab-panel"),
    ("/about", "#about"),
]


@pytest.mark.parametrize("route, ready_selector", _PAGES, ids=[p[0] for p in _PAGES])
def test_page_has_zero_serious_or_critical_axe_violations(
    page: Any, axe_source: str, route: str, ready_selector: str
) -> None:
    """Each main page renders with zero serious/critical accessibility issues."""
    page.goto(route, wait_until="domcontentloaded")
    page.wait_for_selector(ready_selector, state="attached", timeout=5000)
    _assert_axe_clean(page, axe_source, surface=route)


# ── About modal (HTMX-swapped over a tab) ───────────────────────────────────


def test_about_modal_has_zero_serious_or_critical_axe_violations(
    page: Any, axe_source: str
) -> None:
    """Opening the About modal over /search keeps the page axe-clean."""
    page.goto("/search", wait_until="domcontentloaded")
    wait_for_alpine(page)
    page.locator("a.about").click()
    page.locator("#about").wait_for(state="visible", timeout=5000)
    _assert_axe_clean(page, axe_source, surface="/search + About modal")


# ── Command palette (⌘K) ─────────────────────────────────────────────────────


def test_command_palette_has_zero_serious_or_critical_axe_violations(
    page: Any, axe_source: str
) -> None:
    """The command palette, opened via ⌘K, audits clean while visible."""
    page.goto("/search", wait_until="domcontentloaded")
    wait_for_alpine(page)
    page.keyboard.press("Control+k")
    page.locator("#palette").wait_for(state="visible", timeout=5000)
    _assert_axe_clean(page, axe_source, surface="/search + command palette")


# ── Keyboard-help overlay (?) ────────────────────────────────────────────────


def test_help_overlay_has_zero_serious_or_critical_axe_violations(
    page: Any, axe_source: str
) -> None:
    """The keyboard-help overlay, opened via ?, audits clean while visible."""
    page.goto("/search", wait_until="domcontentloaded")
    wait_for_alpine(page)
    # ``?`` is Shift+/ — the bare-key handler in mojica.js toggles Help.
    page.keyboard.press("Shift+?")
    page.locator("#help").wait_for(state="visible", timeout=5000)
    _assert_axe_clean(page, axe_source, surface="/search + help overlay")
