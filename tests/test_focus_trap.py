"""U3 — focus management (trap + restore) for modal overlays.

Runtime focus behaviour (Tab wrapping, restore-to-trigger) can only be
exercised in a real browser; that is the job of the U5 / Playwright a11y
gate. What pytest CAN pin — and what this module pins — is the *structural*
contract the trap depends on:

  * the vendored ``focus_trap.js`` utility is served and exports the public
    surface + carries the focusable-selector, wrap, and restore primitives;
  * ``base.html`` (hence every full-page tab) loads the script;
  * each of the three real *modal* overlays — the command palette, the
    keyboard-help overlay, and the About modal — carries
    ``role="dialog"`` + ``aria-modal="true"`` AND the ``data-focus-trap``
    hook the utility keys off;
  * the anchored, non-modal toolbar popovers (``role="dialog"`` WITHOUT
    ``aria-modal``, dismissed by click-outside) are NOT tagged — trapping a
    transient disclosure widget is wrong UX and not a WCAG requirement, so
    the negative control guards against over-application.

Hermetic: rendered HTML + static JS text only, no heavy models.
"""

from __future__ import annotations

from pathlib import Path

JS = Path("web/static/js/focus_trap.js").read_text(encoding="utf-8")


# ── The vendored utility ──────────────────────────────────────────────────


def test_focus_trap_js_served(client) -> None:
    """``/static/js/focus_trap.js`` is reachable via the static mount and
    exposes the ``FocusTrap`` public surface (mirrors ToastBus / Palette).

    The asset is a vendored vanilla IIFE (no build step), so substring
    assertions on the source are valid evidence the served file is the U3
    utility and not an empty placeholder.
    """
    r = client.get("/static/js/focus_trap.js")
    assert r.status_code == 200, r.text[:200]
    body = r.text
    assert "window.FocusTrap" in body
    # The activate/deactivate pair is the lifecycle contract.
    assert "activate" in body
    assert "deactivate" in body


def test_focus_trap_js_has_focusable_selector() -> None:
    """The focusable-element selector covers the standard set so the wrap
    logic walks the same controls a browser's Tab order would.

    Pin a representative slice (links, enabled buttons/inputs, and the
    ``[tabindex]:not([tabindex="-1"])`` clause that includes custom
    focusables while excluding programmatic-only ones). If a refactor
    narrows the selector, Tab could skip real controls — fail loudly.
    """
    assert "FOCUSABLE" in JS
    assert "a[href]" in JS
    assert "button:not([disabled])" in JS
    assert 'input:not([disabled]):not([type="hidden"])' in JS
    # The clause that admits custom focusables but excludes tabindex=-1.
    assert '[tabindex]:not([tabindex="-1"])' in JS


def test_focus_trap_js_has_wrap_logic() -> None:
    """The Tab handler wraps at both ends: forward from the last focusable
    to the first, backward (Shift+Tab) from the first to the last.

    Asserted via the structural markers (Tab key gate, shiftKey branch,
    first/last endpoints, and preventDefault on the wrap) rather than
    behaviour, which is U5's remit.
    """
    assert "'Tab'" in JS or '"Tab"' in JS
    assert "shiftKey" in JS
    # The two wrap endpoints.
    assert "first" in JS and "last" in JS
    # A wrapped Tab is consumed so it does not also move focus natively.
    assert "preventDefault" in JS


def test_focus_trap_js_has_restore_logic() -> None:
    """On open the trap captures the trigger (``document.activeElement``);
    on close it returns focus to it.

    Pin the capture (activeElement read), the stored-trigger field, and the
    guarded restore (``.focus()`` only when the trigger is still connected).
    """
    assert "document.activeElement" in JS
    assert "__focusTrapTrigger" in JS
    # Restore is guarded on the trigger still being in the document.
    assert "isConnected" in JS
    assert ".focus()" in JS


def test_focus_trap_js_observes_visibility() -> None:
    """The utility is visibility-driven (decoupled from Alpine / HTMX): a
    MutationObserver reconciles each ``[data-focus-trap]`` element, and the
    ``getClientRects`` predicate is what makes the fixed-position backdrops
    detectable (``offsetParent`` is null for them)."""
    assert "MutationObserver" in JS
    assert "data-focus-trap" in JS
    assert "getClientRects" in JS


# ── base.html wiring ──────────────────────────────────────────────────────


def test_base_shell_loads_focus_trap_js(client) -> None:
    """Every full-page render ships the ``focus_trap.js`` <script> (it lives
    in ``base.html``), so the trap is armed on every tab.

    Pinned on /search but the tag is in the shared chrome head, so any
    full-page route carries it.
    """
    r = client.get("/search")
    assert r.status_code == 200
    assert "/static/js/focus_trap.js" in r.text


# ── The three modal overlays carry the dialog + trap contract ─────────────


def _assert_modal_contract(html: str, anchor_id: str) -> None:
    """The element identified by ``anchor_id`` is a trapping modal dialog:
    ``role="dialog"`` + ``aria-modal="true"`` + ``data-focus-trap`` all sit
    on the same opening tag."""
    needle = f'id="{anchor_id}"'
    assert needle in html, f"missing #{anchor_id}"
    # Slice the opening tag of the overlay container and assert the trio.
    start = html.index(needle)
    tag_start = html.rfind("<", 0, start)
    tag_end = html.index(">", start)
    tag = html[tag_start:tag_end]
    assert 'role="dialog"' in tag, f"#{anchor_id}: missing role=dialog"
    assert 'aria-modal="true"' in tag, f"#{anchor_id}: missing aria-modal"
    assert "data-focus-trap" in tag, f"#{anchor_id}: missing data-focus-trap hook"


def test_palette_overlay_is_trapping_dialog(client) -> None:
    """The command palette (#palette) is a focus-trapping modal dialog."""
    r = client.get("/search")
    assert r.status_code == 200
    _assert_modal_contract(r.text, "palette")


def test_help_overlay_is_trapping_dialog(client) -> None:
    """The keyboard-help overlay (#help) is a focus-trapping modal dialog."""
    r = client.get("/search")
    assert r.status_code == 200
    _assert_modal_contract(r.text, "help")


def test_about_modal_is_trapping_dialog(client) -> None:
    """The About modal (#about), served as the HTMX partial, is a
    focus-trapping modal dialog."""
    r = client.get("/api/about")
    assert r.status_code == 200
    _assert_modal_contract(r.text, "about")


def test_about_standalone_page_loads_focus_trap(client) -> None:
    """The JS-off ``/about`` fallback renders the modal inline AND ships the
    trap script, so the dialog still pulls focus on that surface."""
    r = client.get("/about")
    assert r.status_code == 200
    assert "/static/js/focus_trap.js" in r.text
    _assert_modal_contract(r.text, "about")


# ── Negative control: anchored toolbar popovers are NOT trapped ───────────


def test_inline_popovers_are_not_focus_trapped(client) -> None:
    """The Buscar knob-row popovers are ``role="dialog"`` *non-modal*
    disclosure widgets (anchored, ``aria-haspopup="dialog"``, dismissed by
    click-outside). They must NOT carry ``aria-modal`` or ``data-focus-trap``
    — trapping a transient toolbar popover is wrong UX and not required by
    WCAG. This guards against over-applying the trap to every ``role=dialog``.
    """
    r = client.get("/tab/search")
    assert r.status_code == 200
    html = r.text
    # The toolbar popovers exist and advertise themselves as non-modal
    # disclosure (haspopup) anchored controls …
    assert 'aria-haspopup="dialog"' in html
    assert 'class="popover"' in html
    # … and the popover container does not get the modal trap hook. We assert
    # the only ``data-focus-trap`` occurrences on the page belong to the
    # three modal overlays (palette / help; About is injected separately).
    # The popover blocks live inside ``.knob-popover`` wrappers; none should
    # carry the hook.
    for chunk in html.split('class="popover"')[1:]:
        head = chunk[: chunk.index(">")] if ">" in chunk else chunk
        assert "data-focus-trap" not in head
