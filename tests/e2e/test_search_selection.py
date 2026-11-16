"""Scene selection made in Search survives tab changes.

Read-only: the tests search, tick checkboxes and navigate; the selection lives
in ``sessionStorage`` and nothing is written to the archive. Skips when no
query returns two cards.
"""

from __future__ import annotations

from typing import Any

import pytest

pytestmark = pytest.mark.e2e

_QUERIES = ("mar", "street", "homem")
_BAR = ".selection-bar"
_RIMAS_TAB = "a.tab[data-tab='rimas']"
_SEARCH_TAB = "a.tab[data-tab='buscar']"


def _search(page: Any, q: str) -> int:
    page.fill("#search-input", q)
    page.press("#search-input", "Enter")
    try:
        page.wait_for_selector(".b-card", timeout=15000)
    except Exception:
        return 0
    return page.locator(".b-card").count()


def _pick_two(page: Any) -> tuple[str, list[int]]:
    """Return (query, card indexes), preferring two different films."""
    for q in _QUERIES:
        page.goto("/search")
        n = _search(page, q)
        if n < 2:
            continue
        slugs = [page.locator(".b-card").nth(i).get_attribute("data-film-slug") for i in range(n)]
        first = 0
        other = next((i for i, s in enumerate(slugs) if s != slugs[0]), 1)
        return q, [first, other]
    pytest.skip("no query returned two cards")


def _ticked(page: Any) -> int:
    return page.locator(".b-card.checked").count()


def test_selection_survives_tabs_and_opens_in_scenes(page: Any) -> None:
    q, idx = _pick_two(page)
    inspector_calls: list[str] = []
    page.on("request", lambda r: inspector_calls.append(r.url) if "/inspector" in r.url else None)
    for i in idx:
        page.locator(".b-card").nth(i).locator(".check").click()
    assert page.locator(f"{_BAR} .sel-count span").inner_text() == "2"
    page.wait_for_timeout(500)
    assert inspector_calls == []
    keys = [
        page.locator(".b-card").nth(i).get_attribute("data-film-slug")
        + "/"
        + page.locator(".b-card").nth(i).get_attribute("data-scene-id")
        for i in idx
    ]

    page.click(_RIMAS_TAB)
    page.wait_for_load_state("networkidle")
    page.click(_SEARCH_TAB)
    page.wait_for_load_state("networkidle")
    _search(page, q)
    assert _ticked(page) == 2
    assert page.locator(f"{_BAR} .sel-count span").inner_text() == "2"

    page.locator(f"{_BAR} .sel-btn--primary").click()
    page.wait_for_function("location.pathname === '/scenes' && location.search.startsWith('?sel=')")
    page.wait_for_selector(".scenecard")
    assert page.locator(".scenecard").count() == 2
    assert page.locator(".scenecard.checked").count() == 2
    assert all(k.split("/")[1] in page.url for k in keys)

    page.locator(f"{_BAR} .sel-btn--ghost").click()
    page.wait_for_selector(_BAR, state="hidden")
    assert page.evaluate("sessionStorage.getItem('kuaa.selection')") in (None, "", "[]")
