"""Tests for the sidebar film selector partial (T10).

The film selector is a <select> rendered inside the sidebar on every
full-page route (/scenes, /search, /annotate, /processing).  These
tests assert:

  1. The selector markup appears on every tab page when films are
     registered in the library.
  2. The "Acervo inteiro" (aggregate) option is selected when no
     ?film= query param is given.
  3. The matching film option is marked ``selected`` when ?film=<slug>
     is given.
  4. The selector is hidden (not rendered) when the library is empty.

Full-page routes inject ``films`` and ``current_slug`` via T9; the
selector template gates on ``{% if films %}`` so an empty library
suppresses it gracefully.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# ── Two-film fixture (mirrors test_routes_multi_film._seed_film) ──────────────

def _seed_film(library_dir: Path, slug: str, title: str, scene_ids: list[int]) -> None:
    """Register a film and create its per-film directory layout."""
    from cinemateca.library import register_film

    register_film(
        library_dir,
        slug=slug,
        title=title,
        year=None,
        raw_filename=f"{slug}.mp4",
    )
    film_dir = library_dir / slug
    (film_dir / "raw").mkdir(parents=True, exist_ok=True)
    (film_dir / "raw" / f"{slug}.mp4").touch()
    (film_dir / "frames" / "keyframes").mkdir(parents=True, exist_ok=True)
    (film_dir / "embeddings").mkdir(parents=True, exist_ok=True)
    meta = film_dir / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    kf_meta = [
        {
            "scene_id": sid,
            "filepath": f"library/{slug}/frames/keyframes/{sid}.jpg",
            "timecode_start": f"00:{sid:02d}:00",
            "start_time_s": float(sid * 60),
            "end_time_s": float(sid * 60 + 10),
        }
        for sid in scene_ids
    ]
    (meta / "keyframes_metadata.json").write_text(json.dumps(kf_meta))
    (meta / "scene_tags.json").write_text(json.dumps({"outdoor": scene_ids}))


@pytest.fixture()
def two_film_client(tmp_config, client):
    """TestClient with two registered films: film_a and film_b."""
    library_dir = Path(tmp_config.paths.library_dir)
    _seed_film(library_dir, slug="film_a", title="Film A", scene_ids=[1, 2, 3])
    _seed_film(library_dir, slug="film_b", title="Film B", scene_ids=[4, 5])
    return client


# ── Selector presence on full-page routes ────────────────────────────────────

FULL_PAGE_ROUTES = ["/scenes", "/search", "/annotate", "/processing"]


class TestFilmSelectorPresence:
    """The selector appears on every tab page when films are registered."""

    @pytest.mark.parametrize("path", FULL_PAGE_ROUTES)
    def test_selector_present_on_every_tab(self, two_film_client, path):
        """Full-page GET renders the film-selector widget in the sidebar."""
        r = two_film_client.get(path)
        assert r.status_code == 200, r.text[:300]
        assert 'id="film-select"' in r.text, (
            f"film selector not found in {path} response"
        )

    @pytest.mark.parametrize("path", FULL_PAGE_ROUTES)
    def test_selector_lists_both_films(self, two_film_client, path):
        """Both registered film titles appear as <option> values."""
        r = two_film_client.get(path)
        assert "Film A" in r.text
        assert "Film B" in r.text

    @pytest.mark.parametrize("path", FULL_PAGE_ROUTES)
    def test_aggregate_option_present(self, two_film_client, path):
        """The 'Acervo inteiro' aggregate option is always present."""
        r = two_film_client.get(path)
        # The option value="" is the aggregate sentinel.
        assert 'value=""' in r.text or "Acervo inteiro" in r.text


# ── Current-slug selection ────────────────────────────────────────────────────

class TestFilmSelectorSelection:
    """The correct <option> is marked selected based on ?film=."""

    def test_no_slug_selects_aggregate(self, two_film_client):
        """No ?film= → the empty-value option (Acervo inteiro) is selected.

        Uses a regex so the assertion is tied to THIS option specifically
        (not just "any option is selected somewhere") — the template
        renders the selected attr on a new line after value="".
        """
        import re
        r = two_film_client.get("/scenes")
        assert r.status_code == 200
        assert re.search(r'value=""\s+selected', r.text), (
            "Aggregate (value='') option must carry the selected attribute"
        )

    def test_film_a_slug_selects_film_a_option(self, two_film_client):
        """?film=film_a → the film_a option is marked selected (and ONLY it)."""
        import re
        r = two_film_client.get("/scenes?film=film_a")
        assert r.status_code == 200
        assert re.search(r'value="film_a"\s+selected', r.text), (
            "film_a option must carry the selected attribute when ?film=film_a"
        )
        # And the aggregate option must NOT be selected.
        assert not re.search(r'value=""\s+selected', r.text), (
            "Aggregate option must not be selected when ?film=film_a"
        )

    def test_film_b_slug_selects_film_b_option(self, two_film_client):
        """?film=film_b → the film_b option is marked selected (and ONLY it)."""
        import re
        r = two_film_client.get("/scenes?film=film_b")
        assert r.status_code == 200
        assert re.search(r'value="film_b"\s+selected', r.text), (
            "film_b option must carry the selected attribute when ?film=film_b"
        )
        assert not re.search(r'value="film_a"\s+selected', r.text), (
            "film_a option must not be selected when ?film=film_b"
        )

    def test_search_tab_respects_current_slug(self, two_film_client):
        """?film= param is respected on /search too, not just /scenes."""
        import re
        r = two_film_client.get("/search?film=film_a")
        assert r.status_code == 200
        assert re.search(r'value="film_a"\s+selected', r.text)

    def test_annotate_tab_respects_current_slug(self, two_film_client):
        """?film= param is respected on /annotate too."""
        import re
        r = two_film_client.get("/annotate?film=film_b")
        assert r.status_code == 200
        assert re.search(r'value="film_b"\s+selected', r.text)


# ── Empty library suppresses the selector ────────────────────────────────────

class TestFilmSelectorEmpty:
    """The selector widget is hidden when no films are registered."""

    def test_selector_absent_when_no_films(self, client):
        """Empty library → no film-selector in the sidebar.

        The ``client`` fixture has no films registered (tmp_config, no seed).
        The partial gates on ``{% if films %}`` so the <select> must be absent.
        """
        r = client.get("/scenes")
        assert r.status_code == 200
        assert 'id="film-select"' not in r.text

    def test_page_still_renders_when_no_films(self, client):
        """No films does not crash any full-page route."""
        for path in FULL_PAGE_ROUTES:
            r = client.get(path)
            assert r.status_code == 200, f"{path} returned {r.status_code}"
            assert "<!DOCTYPE html>" in r.text
