"""Film and scene carry from Search to the other tabs.

A Search result card loads the inspector with HTMX and never changes the URL,
so the top-bar tab links have nothing to read the selection from. The page
remembers it in ``sessionStorage`` (``kuaa.context``) instead.

Read-only: the tests search, click and navigate. Skips when the library has
fewer than two films or no query returns a card from a non-first film.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

from .conftest import REPO_ROOT

pytestmark = pytest.mark.e2e

_QUERIES = ("mar", "street", "homem")
_SCENES_TAB = "a.tab[data-tab='cenas']"
_ANNOTATE_TAB = "a.tab[data-tab='anotar']"


def _library_slugs() -> list[str]:
    lib = REPO_ROOT / "data" / "library"
    if not lib.is_dir():
        return []
    return sorted(p.name for p in lib.iterdir() if (p / "embeddings").is_dir())


def _query(page: Any) -> dict[str, list[str]]:
    return parse_qs(urlparse(page.url).query)


def _select_non_first_card(page: Any) -> tuple[str, str]:
    """Search, click a card from a non-first film; return (slug, scene id)."""
    slugs = _library_slugs()
    if len(slugs) < 2:
        pytest.skip("library has fewer than two processed films")
    for q in _QUERIES:
        page.goto("/search")
        page.fill("#search-input", q)
        page.press("#search-input", "Enter")
        try:
            page.wait_for_selector(".b-card", timeout=15000)
        except Exception:
            continue
        cards = page.locator(".b-card")
        for i in range(cards.count()):
            card = cards.nth(i)
            slug = card.get_attribute("data-film-slug") or ""
            if slug and slug != slugs[0]:
                scene = card.get_attribute("data-scene-id") or ""
                with page.expect_response(lambda r: "/inspector" in r.url):
                    card.click()
                page.wait_for_load_state("networkidle")
                return slug, scene
    pytest.skip("no query returned a card from a non-first film")


def test_scenes_tab_follows_the_selected_search_card(page: Any) -> None:
    slug, scene = _select_non_first_card(page)
    page.click(_SCENES_TAB)
    page.wait_for_url("**/scenes*")
    q = _query(page)
    assert q.get("film") == [slug]
    assert q.get("scene") == [scene]


def test_all_films_clears_the_remembered_context(page: Any) -> None:
    _select_non_first_card(page)
    page.click(".ch-lp .ch-film--all")
    page.wait_for_load_state("networkidle")
    page.click(_SCENES_TAB)
    page.wait_for_url("**/scenes*")
    assert "film" not in _query(page)
    assert page.evaluate("sessionStorage.getItem('kuaa.context')") is None


def test_url_film_wins_over_the_remembered_one(page: Any) -> None:
    slugs = _library_slugs()
    _select_non_first_card(page)
    other = slugs[0]
    page.goto(f"/scenes?film={other}")
    page.click(_ANNOTATE_TAB)
    page.wait_for_url("**/annotate*")
    assert _query(page).get("film") == [other]
