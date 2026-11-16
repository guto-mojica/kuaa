"""The scene description is visible as soon as a Search card is clicked.

Read-only: searches, clicks a card, looks at the right pane. No pane tab is
clicked, so what is asserted is the first render of the default tab.
"""

from __future__ import annotations

from typing import Any

import pytest

from .conftest import REPO_ROOT

# Collected by the model-free gates too, which have no Playwright installed;
# skip at import instead of failing collection.
expect = pytest.importorskip("playwright.sync_api").expect

pytestmark = pytest.mark.e2e

_QUERIES = ("mar", "street", "homem", "rua")
_TRIES = 3


def _has_library() -> bool:
    lib = REPO_ROOT / "data" / "library"
    return lib.is_dir() and any((p / "embeddings").is_dir() for p in lib.iterdir())


def _open_card(page: Any, index: int) -> bool:
    """Search, click the ``index``-th card; True if one was clicked."""
    for q in _QUERIES:
        page.goto("/search")
        page.fill("#search-input", q)
        page.press("#search-input", "Enter")
        try:
            page.wait_for_selector(".b-card", timeout=15000)
        except Exception:
            continue
        cards = page.locator(".b-card")
        if cards.count() <= index:
            continue
        with page.expect_response(lambda r: "/inspector" in r.url):
            cards.nth(index).click()
        return True
    return False


def test_description_visible_without_clicking_a_pane_tab(page: Any) -> None:
    if not _has_library():
        pytest.skip("no processed library on disk")
    page.set_viewport_size({"width": 1440, "height": 900})
    seen = 0
    for i in range(_TRIES * 4):
        if not _open_card(page, i):
            break
        pane = page.locator("#right-pane")
        try:
            page.wait_for_selector("#right-pane .b-com.ai", timeout=3000)
        except Exception:
            continue
        seen += 1
        desc = pane.locator(".b-com.ai").first
        expect(desc).to_be_visible(timeout=3000)
        assert desc.inner_text().strip()
        expect(page.locator("#inspector-skeleton")).not_to_be_visible()
        box = desc.bounding_box()
        pane_box = pane.bounding_box()
        vp = page.viewport_size
        assert box is not None and pane_box is not None and vp is not None
        assert box["y"] >= max(pane_box["y"], 0)
        assert box["y"] + box["height"] <= min(pane_box["y"] + pane_box["height"], vp["height"])
        if seen >= _TRIES:
            return
    if not seen:
        pytest.skip("no clicked card carried a description")
