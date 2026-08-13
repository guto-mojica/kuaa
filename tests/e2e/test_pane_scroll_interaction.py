"""E2E: the inspector panes must actually be able to scroll.

``tests/test_web_routes.py`` and the snapshot suites pin the inspector
MARKUP, and the CSS files declare ``overflow-y: auto`` on every pane's inner
scroll region. Neither can see whether that declaration does anything — and
for a long time it did not.

The regression this module guards: a scroll region only scrolls when its
parent chain hands it a DEFINITE height. Two things broke that for the
Scenes and Buscar inspectors:

  1. ``.c-rp .inner`` / ``.b-rp .inner`` are ``flex: 1`` items, and a flex
     item's ``min-height`` defaults to ``auto``, which refuses to shrink the
     item below its content height. Without ``min-height: 0`` the box grows
     to fit its content, so ``overflow-y: auto`` has nothing to overflow.
  2. Those two tabs mount the pane as
     ``.rp-slot > #right-pane > section.c-rp`` — two PLAIN BLOCK wrappers
     between ``.tab-panel``'s flex line and the styled pane (``.rp-slot`` is
     the skeleton's positioning context; the bare aside is a neutral
     hx-swap="innerHTML" target). A block does not pass its height down, so
     the pane sized to its content and the fix in (1) had nothing to bite on.

Symptom in the product: a long LLM description was clipped with no
scrollbar, and the inspector's action buttons — the LAST child of
``.inner`` — were cut off entirely and unreachable at any pane width.

Both are measured here rather than asserted from CSS, because both were
"fixed" twice from static reading before a browser showed what was actually
happening. ``scrollHeight > clientHeight`` proves a scrollbar exists;
driving ``scrollTop`` and re-measuring proves it moves and brings the
buttons into view.

Library-dependent: the long-description scene lives in the real demo
archive, so the content-overflow assertions skip when it is absent (a
machine holding a different library still runs the structural checks).
"""

from __future__ import annotations

from typing import Any

import pytest

pytestmark = pytest.mark.e2e

# Jeca Tatu scene 2 carries the archive's longest single description
# (~845 chars), which is what pushes .inner past the pane height at a
# 900px viewport. Structural checks below do not depend on it.
_LONG_DESC_FILM = "jeca_tatu_1959"
_LONG_DESC_SCENE = 2

_MEASURE = """
([paneSel, scrollSel]) => {
  const pane = document.querySelector(paneSel);
  const scroller = document.querySelector(scrollSel);
  if (!pane || !scroller) return {missing: {pane: !!pane, scroller: !!scroller}};
  const cs = getComputedStyle(pane);
  const ss = getComputedStyle(scroller);
  return {
    paneClientH: pane.clientHeight,
    paneFlex: cs.flex,
    paneMinHeight: cs.minHeight,
    scrollerClientH: scroller.clientHeight,
    scrollerScrollH: scroller.scrollHeight,
    scrollerOverflowY: ss.overflowY,
    scrollerMinHeight: ss.minHeight,
    canScroll: scroller.scrollHeight > scroller.clientHeight + 1,
  };
}
"""


def _measure(page: Any, pane_sel: str, scroll_sel: str) -> dict:
    return page.evaluate(_MEASURE, [pane_sel, scroll_sel])


def test_scenes_inspector_pane_gets_a_definite_height(page: Any) -> None:
    """``#right-pane > .c-rp`` must be height-constrained, not content-sized.

    This is the wrapper-chain half of the bug. If either ``.rp-slot`` or
    ``#right-pane`` stops conducting height, the pane's clientHeight
    collapses toward its content and this fails — even if the CSS still
    says ``overflow-y: auto`` further in.
    """
    page.set_viewport_size({"width": 1440, "height": 900})
    page.goto("/scenes")
    page.wait_for_selector("#right-pane > .c-rp")
    m = _measure(page, "#right-pane > .c-rp", "#right-pane .c-rp .inner")
    assert "missing" not in m, m
    # The pane fills the tab body (viewport 900 minus topbar), so it is far
    # taller than any plausible content-sized collapse.
    assert m["paneClientH"] > 500, f"pane not stretched to the tab body: {m}"
    assert m["scrollerMinHeight"] == "0px", f"inner lost min-height:0: {m}"
    assert m["scrollerOverflowY"] == "auto", f"inner is not a scroll region: {m}"
    # The scroller must be SHORTER than its pane (header takes the rest);
    # equal-or-taller means it escaped the constraint again.
    assert m["scrollerClientH"] <= m["paneClientH"], m


def test_buscar_inspector_pane_gets_a_definite_height(page: Any) -> None:
    """Same wrapper chain as Scenes (``.rp-slot > #right-pane > .b-rp``)."""
    page.set_viewport_size({"width": 1440, "height": 900})
    page.goto("/search")
    page.wait_for_selector("#right-pane > .b-rp")
    pane = page.evaluate(
        """() => {
            const p = document.querySelector('#right-pane > .b-rp');
            return {h: p.clientHeight, flex: getComputedStyle(p).flex};
        }"""
    )
    assert pane["h"] > 500, f"buscar pane not stretched: {pane}"


def test_scenes_inspector_scrolls_a_long_description_to_its_buttons(page: Any) -> None:
    """A long description must produce a real scrollbar, and scrolling it
    must bring the action buttons (last child of ``.inner``) into view.

    This is the user-visible contract: before the fix the buttons
    ("Buscar semelhantes" / "Anotar" / "Buscar rimas visuais") were clipped
    away with no way to reach them.
    """
    page.set_viewport_size({"width": 1440, "height": 900})
    page.goto(f"/scenes?film={_LONG_DESC_FILM}&scene={_LONG_DESC_SCENE}")
    try:
        page.wait_for_selector("#right-pane .c-rp .actions", timeout=8000)
    except Exception:  # pragma: no cover - depends on the local archive
        pytest.skip(f"{_LONG_DESC_FILM} scene {_LONG_DESC_SCENE} not in this library")

    m = _measure(page, "#right-pane > .c-rp", "#right-pane .c-rp .inner")
    if not m.get("canScroll"):
        pytest.skip(f"description does not overflow at this viewport: {m}")

    # The buttons start below the visible area — that is the overflow.
    geom = page.evaluate(
        """() => {
            const inner = document.querySelector('#right-pane .c-rp .inner');
            const acts = document.querySelector('#right-pane .c-rp .actions');
            const before = acts.getBoundingClientRect().bottom
                <= inner.getBoundingClientRect().bottom + 1;
            inner.scrollTop = inner.scrollHeight;   // drive the scrollbar
            const i = inner.getBoundingClientRect();
            const a = acts.getBoundingClientRect();
            return {
                visibleBefore: before,
                scrollTop: inner.scrollTop,
                visibleAfter: a.bottom <= i.bottom + 1 && a.top >= i.top - 1,
                labels: [...acts.querySelectorAll('a,button')].map(
                    (e) => e.textContent.trim()
                ),
            };
        }"""
    )
    assert geom["scrollTop"] > 0, f"pane did not scroll: {geom}"
    assert geom["visibleAfter"], f"actions still not reachable after scrolling: {geom}"
    assert geom["labels"], "inspector rendered no action buttons"
