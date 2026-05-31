"""E2E focus-management runtime tests (U3) — the carry-in browser verification.

``tests/test_focus_trap.py`` pins the *structural* contract (the vendored
``focus_trap.js`` surface, the ``role=dialog`` + ``aria-modal`` + ``data-focus-trap``
trio on the three modal overlays, the non-modal popover negative control).
What it explicitly defers to "the U5 / Playwright a11y gate" is the *runtime*
behaviour — and that is what this module drives in a real browser:

  * **⌘K → focus lands on ``#cp-input``** (the palette pulls focus on open).
  * **Tab at the last focusable wraps to the first**, and **Shift+Tab at the
    first wraps to the last**, inside a modal (exercised on the About modal,
    which has many focusables: close button → repo links → footer Close).
  * **Esc closes the palette AND restores focus to the trigger.**
  * **The About modal's close (HTMX-delete of the node) restores focus** to the
    About link that opened it.
  * **A non-modal toolbar ``.popover`` is NOT trapped** — Tab moves focus out of
    it to the next control on the page (the over-application negative control).

Focus moves are deferred one frame inside ``focus_trap.js`` (``setTimeout(0)``
after the visibility flip) and driven by a MutationObserver, so every assertion
polls for the settled focus state via ``wait_for_function`` rather than reading
``activeElement`` on the same tick as the trigger.
"""

from __future__ import annotations

from typing import Any

import pytest

from .conftest import wait_for_alpine

pytestmark = pytest.mark.e2e


def _active_id(page: Any) -> str:
    """The id of the currently-focused element (``''`` if none / unset)."""
    return page.evaluate("() => (document.activeElement && document.activeElement.id) || ''")


def _wait_active_id(page: Any, expected: str, *, timeout: int = 5000) -> None:
    """Poll until ``document.activeElement.id === expected``.

    Used instead of an immediate assert because the trap defers its focus
    move one animation frame after the overlay becomes visible.
    """
    page.wait_for_function(
        "(want) => (document.activeElement && document.activeElement.id) === want",
        arg=expected,
        timeout=timeout,
    )


def _open_palette(page: Any) -> None:
    """Open the command palette via ⌘K and wait for it to be visible + focused."""
    wait_for_alpine(page)
    page.keyboard.press("Control+k")
    page.locator("#palette").wait_for(state="visible", timeout=5000)


# ── ⌘K focus + Esc-restore (palette) ──────────────────────────────────────


def test_cmdk_opens_palette_and_focuses_input(page: Any) -> None:
    """Pressing ⌘K opens the palette and moves focus to ``#cp-input``."""
    page.goto("/search", wait_until="domcontentloaded")
    _open_palette(page)
    _wait_active_id(page, "cp-input")
    assert _active_id(page) == "cp-input"


def test_palette_esc_closes_and_restores_focus_to_trigger(page: Any) -> None:
    """Esc closes the palette and returns focus to the element that opened it.

    We give the trap a concrete trigger by focusing the page's search box
    first; the trap captures ``document.activeElement`` at open time, so on
    close it must restore focus there (not leave it stranded on the now-hidden
    palette input).
    """
    page.goto("/search", wait_until="domcontentloaded")
    # Establish a real trigger: focus the main search input.
    page.locator("#search-input").focus()
    _wait_active_id(page, "search-input")

    _open_palette(page)
    _wait_active_id(page, "cp-input")

    # Esc dismisses (Alpine @keydown.escape.window on #palette).
    page.keyboard.press("Escape")
    page.locator("#palette").wait_for(state="hidden", timeout=5000)

    # Focus restored to the trigger.
    _wait_active_id(page, "search-input")
    assert _active_id(page) == "search-input"


# ── Tab wrap at both ends (About modal — many focusables) ──────────────────


def _open_about(page: Any) -> None:
    """Open the About modal via its HTMX link; wait for the dialog + trap."""
    wait_for_alpine(page)
    page.locator("a.about").click()
    page.locator("#about").wait_for(state="visible", timeout=5000)
    # The trap moves focus inside on open; wait until activeElement is within
    # the dialog before exercising Tab wrap.
    page.wait_for_function(
        "() => { const d = document.getElementById('about');"
        " return d && d.contains(document.activeElement); }",
        timeout=5000,
    )


def test_about_modal_tab_wraps_forward_from_last_to_first(page: Any) -> None:
    """Tab from the last focusable in the modal wraps to the first.

    The trap computes the focusable list in DOM order; we move focus onto the
    last one explicitly, press Tab, and assert focus wrapped to the first.
    """
    page.goto("/search", wait_until="domcontentloaded")
    _open_about(page)

    # Resolve first/last focusable via the trap's own public helper so the test
    # walks the exact same list the wrap logic does.
    handles = page.evaluate_handle(
        "() => window.FocusTrap.focusable(document.getElementById('about'))"
    )
    count = page.evaluate("(list) => list.length", handles)
    assert count >= 2, f"About modal should expose multiple focusables, got {count}"

    # Focus the LAST focusable, then Tab → should wrap to the FIRST.
    page.evaluate("(list) => list[list.length - 1].focus()", handles)
    page.wait_for_function(
        "(list) => document.activeElement === list[list.length - 1]",
        arg=handles,
        timeout=5000,
    )
    page.keyboard.press("Tab")
    page.wait_for_function(
        "(list) => document.activeElement === list[0]",
        arg=handles,
        timeout=5000,
    )
    assert page.evaluate("(list) => document.activeElement === list[0]", handles)


def test_about_modal_shift_tab_wraps_backward_from_first_to_last(page: Any) -> None:
    """Shift+Tab from the first focusable wraps to the last."""
    page.goto("/search", wait_until="domcontentloaded")
    _open_about(page)

    handles = page.evaluate_handle(
        "() => window.FocusTrap.focusable(document.getElementById('about'))"
    )
    count = page.evaluate("(list) => list.length", handles)
    assert count >= 2

    # Focus the FIRST focusable, then Shift+Tab → should wrap to the LAST.
    page.evaluate("(list) => list[0].focus()", handles)
    page.wait_for_function(
        "(list) => document.activeElement === list[0]",
        arg=handles,
        timeout=5000,
    )
    page.keyboard.press("Shift+Tab")
    page.wait_for_function(
        "(list) => document.activeElement === list[list.length - 1]",
        arg=handles,
        timeout=5000,
    )
    assert page.evaluate("(list) => document.activeElement === list[list.length - 1]", handles)


# ── About modal close restores focus (HTMX-delete path) ────────────────────


def test_about_close_restores_focus_to_trigger(page: Any) -> None:
    """Closing the About modal returns focus to the About link that opened it.

    The About modal is REMOVED from the DOM on close (its Alpine ``close()``
    empties ``#modal-container``), so the trap's restore runs via the
    MutationObserver's ``deactivateDetached`` branch — a different code path
    than the palette's CSS-hide. Both must restore the trigger.
    """
    page.goto("/search", wait_until="domcontentloaded")
    about_link = page.locator("a.about")
    about_link.focus()
    # The About link has no id; assert focus is on the link element itself.
    page.wait_for_function(
        "() => document.activeElement && document.activeElement.classList.contains('about')",
        timeout=5000,
    )

    _open_about(page)

    # Close via the in-modal X button (role=button, aria-label Close).
    page.locator("#about .ab-head button.close").click()
    page.locator("#about").wait_for(state="detached", timeout=5000)

    # Focus restored to the About link (the captured trigger).
    page.wait_for_function(
        "() => document.activeElement && document.activeElement.classList.contains('about')",
        timeout=5000,
    )
    assert page.evaluate(
        "() => !!(document.activeElement && document.activeElement.classList.contains('about'))"
    )


# ── Negative control: non-modal toolbar popover is NOT trapped ─────────────


def test_toolbar_popover_is_not_focus_trapped(page: Any) -> None:
    """A non-modal toolbar ``.popover`` does not trap Tab.

    The Buscar retrieval knobs are anchored, non-modal disclosure widgets
    (``role=dialog`` WITHOUT ``aria-modal``, dismissed by click-outside). They
    must NOT carry the focus trap — Tab from inside one moves on to the next
    control on the page rather than cycling within the popover. We open the
    first knob popover, focus a control inside it, Tab, and assert focus left
    the popover.
    """
    page.goto("/search", wait_until="domcontentloaded")
    wait_for_alpine(page)

    # Open the first knob popover (the Hybrid/retrieval one) by clicking its
    # toggle. Alpine flips x-show on the sibling .popover.
    first_knob = page.locator(".knob-popover").first
    first_knob.locator("button.knob-toggle").click()
    popover = first_knob.locator(".popover")
    popover.wait_for(state="visible", timeout=5000)

    # Sanity: the popover is NOT a focus-trap target (structural — mirrors the
    # unit-test negative control, re-checked live).
    assert popover.get_attribute("data-focus-trap") is None
    assert popover.get_attribute("aria-modal") is None

    # Focus the first radio inside the popover, then Tab. If the popover were
    # (wrongly) trapped, focus would cycle within it; since it is not, focus
    # advances and leaves the popover subtree.
    first_input = popover.locator("input").first
    first_input.focus()
    page.wait_for_function(
        "(el) => document.activeElement === el",
        arg=first_input.element_handle(),
        timeout=5000,
    )
    # Tab several times; assert that at some point focus is OUTSIDE the popover
    # (a trapped widget would keep activeElement inside it forever).
    escaped = False
    for _ in range(8):
        page.keyboard.press("Tab")
        inside = page.evaluate(
            "() => { const pop = document.querySelector('.knob-popover .popover');"
            " return pop ? pop.contains(document.activeElement) : false; }"
        )
        if not inside:
            escaped = True
            break
    assert escaped, "Tab never left the non-modal popover — it appears to be trapped"
