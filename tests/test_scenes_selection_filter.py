"""``?sel=`` narrows the Scenes page to exactly the chosen scenes."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from api.services.scenes._selection import parse_selection
from tests.test_routes_multi_film import _seed_film

_CARD = re.compile(r'<article class="scenecard')


def _cards(html: str) -> list[str]:
    return re.findall(r'<article class="scenecard[^>]*data-scene-id="(\d+)"', html, re.S)


def test_parse_selection_normal() -> None:
    assert parse_selection("film_a/1,film_b/4") == [("film_a", 1), ("film_b", 4)]


def test_parse_selection_lowercases_slug() -> None:
    assert parse_selection("Film_A/2") == [("film_a", 2)]


@pytest.mark.parametrize("bad", ["a", "a/", "/3", "a/x", "../etc/3", "a/-1", "a b/3"])
def test_parse_selection_drops_malformed(bad: str) -> None:
    assert parse_selection(f"{bad},ok/7") == [("ok", 7)]


def test_parse_selection_drops_duplicates_keeping_first() -> None:
    assert parse_selection("a/1,b/2,a/1,A/1") == [("a", 1), ("b", 2)]


def test_parse_selection_caps_at_200() -> None:
    raw = ",".join(f"a/{i}" for i in range(250))
    out = parse_selection(raw)
    assert len(out) == 200
    assert out[0] == ("a", 0)


@pytest.mark.parametrize("raw", [None, ""])
def test_parse_selection_empty(raw: str | None) -> None:
    assert parse_selection(raw) == []


@pytest.fixture()
def two_films(tmp_config, client):
    lib = Path(tmp_config.paths.library_dir)
    _seed_film(lib, slug="film_a", title="Film A", scene_ids=[1, 2, 3])
    _seed_film(lib, slug="film_b", title="Film B", scene_ids=[4, 5])
    return client


def test_scenes_page_shows_only_selected(two_films) -> None:
    r = two_films.get("/scenes?sel=film_a/2,film_b/5")
    assert r.status_code == 200
    assert len(_CARD.findall(r.text)) == 2
    assert sorted(_cards(r.text)) == ["2", "5"]
    assert "selected scenes" in r.text
    assert "Show all" in r.text


def test_scenes_page_ignores_film_param_in_sel_mode(two_films) -> None:
    r = two_films.get("/scenes?film=film_a&sel=film_a/2,film_b/5")
    assert sorted(_cards(r.text)) == ["2", "5"]


def test_garbage_sel_behaves_as_absent(two_films) -> None:
    base = two_films.get("/scenes")
    r = two_films.get("/scenes?sel=garbage")
    assert len(_CARD.findall(r.text)) == len(_CARD.findall(base.text)) == 5
    assert "Show all" not in r.text


def test_api_scenes_fragment_accepts_sel(two_films) -> None:
    r = two_films.get("/api/scenes?sel=film_a/2,film_b/5")
    assert r.status_code == 200
    assert sorted(_cards(r.text)) == ["2", "5"]
