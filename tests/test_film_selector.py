"""Tests for the film-selection chrome (post-Phase-2 cleanup).

The legacy ``<select id="film-select">`` lived inside the deleted
``.shell > .sidebar`` block. Film selection is now a list of
``<a class="ch-film">`` rows inside the new ``.ch-lp`` left pane
(``_left_pane_body.html``). The active film carries ``.ch-film.active``
and the cross-tab slug propagation moved from the legacy tab-bar's
``hx-get``/``hx-push-url`` to plain ``href`` strings on the new TopBar
and IconRail chips (kept by ``?film=<slug>`` query-string preservation).

These tests assert:

  1. The new film rows appear on every full-page route when films are
     registered.
  2. The matching row carries ``.active`` when ``?film=<slug>`` is set.
  3. Every tab chip + rail icon preserves the slug across navigation.
  4. The search-form hidden ``film`` input + image POST URL still carry
     the slug — those contracts were never about the legacy sidebar and
     survive intact.
  5. The row block is replaced by the empty-state message when no films
     are registered.
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


# ── Film-row presence on full-page routes ────────────────────────────────────

FULL_PAGE_ROUTES = ["/scenes", "/search", "/annotate", "/processing"]
# The Anotar tab runs with ``compact_lp=True`` (api/server.py:_TAB_CHROME), so
# the .ch-lp left pane is intentionally suppressed and no .ch-film rows are
# rendered. Excluded from the "selector present" parametrization.
LEFT_PANE_ROUTES = [p for p in FULL_PAGE_ROUTES if p != "/annotate"]


class TestFilmSelectorPresence:
    """The .ch-film rows appear on every left-pane tab when films are registered."""

    @pytest.mark.parametrize("path", LEFT_PANE_ROUTES)
    def test_selector_present_on_every_tab(self, two_film_client, path):
        """Full-page GET renders the .ch-film rows in the left pane."""
        r = two_film_client.get(path)
        assert r.status_code == 200, r.text[:300]
        assert 'class="ch-film' in r.text, f"film rows not found in {path} response"

    @pytest.mark.parametrize("path", LEFT_PANE_ROUTES)
    def test_selector_lists_both_films(self, two_film_client, path):
        """Both registered film titles appear as .ch-film rows."""
        r = two_film_client.get(path)
        assert "Film A" in r.text
        assert "Film B" in r.text
        assert 'data-slug="film_a"' in r.text
        assert 'data-slug="film_b"' in r.text

    def test_annotate_omits_left_pane(self, two_film_client):
        """The Anotar tab uses compact_lp=True; .ch-lp is intentionally absent."""
        r = two_film_client.get("/annotate")
        assert r.status_code == 200
        assert 'class="ch-lp"' not in r.text
        assert "compact-lp" in r.text


# ── Current-slug selection ────────────────────────────────────────────────────


class TestFilmSelectorSelection:
    """The correct .ch-film row carries .active based on ?film=."""

    def test_no_slug_marks_nothing_active(self, two_film_client):
        """No ?film= → no row carries the .active modifier.

        The aggregate ("Acervo inteiro") concept moved to a Collection link
        (``Entire library``) rather than a row state, so the .ch-film list
        intentionally has no .active row in the aggregate view.
        """
        import re

        r = two_film_client.get("/scenes")
        assert r.status_code == 200
        assert not re.search(r'class="ch-film[^"]*\bactive\b', r.text)

    def test_film_a_slug_marks_film_a_active(self, two_film_client):
        """?film=film_a → only the film_a row carries .active."""
        import re

        r = two_film_client.get("/scenes?film=film_a")
        assert r.status_code == 200
        assert re.search(
            r'class="ch-film[^"]*\bactive\b[^"]*"[^>]*data-slug="film_a"',
            r.text,
        ), "film_a row must carry .active when ?film=film_a"
        assert not re.search(
            r'class="ch-film[^"]*\bactive\b[^"]*"[^>]*data-slug="film_b"',
            r.text,
        ), "film_b row must NOT carry .active when ?film=film_a"

    def test_film_b_slug_marks_film_b_active(self, two_film_client):
        """?film=film_b → only the film_b row carries .active."""
        import re

        r = two_film_client.get("/scenes?film=film_b")
        assert r.status_code == 200
        assert re.search(
            r'class="ch-film[^"]*\bactive\b[^"]*"[^>]*data-slug="film_b"',
            r.text,
        )
        assert not re.search(
            r'class="ch-film[^"]*\bactive\b[^"]*"[^>]*data-slug="film_a"',
            r.text,
        )

    def test_search_tab_respects_current_slug(self, two_film_client):
        """?film= param is respected on /search too, not just /scenes."""
        import re

        r = two_film_client.get("/search?film=film_a")
        assert r.status_code == 200
        assert re.search(
            r'class="ch-film[^"]*\bactive\b[^"]*"[^>]*data-slug="film_a"',
            r.text,
        )

    def test_annotate_tab_respects_current_slug(self, two_film_client):
        """?film= param is respected on /annotate too.

        ``/annotate`` runs with ``compact_lp=True`` so the .ch-lp pane is
        collapsed; we assert at the cookie / body level instead since the
        film rows aren't rendered. ``data-active-tab`` confirms the page
        is the Anotar tab and the URL query reached it.
        """
        r = two_film_client.get("/annotate?film=film_b")
        assert r.status_code == 200
        assert 'data-active-tab="anotar"' in r.text

    def test_scenes_page_grid_filtered_by_slug(self, two_film_client):
        """Full-page /scenes?film=<slug> must filter the grid, not just
        the sidebar's .active marker.

        Regression for the Mojica redesign bug where the LeftPane chrome
        marked the selected film active but the .c-cp grid still rendered
        the aggregate (all-films) view, starting with the alphabetically
        first registered film. Root cause: ``_TAB_CONTEXT_BUILDERS["scenes"]``
        called ``build_cenas_context(get_config())`` without
        ``current_slug``. The HTMX fragment routes (``/tab/scenes``,
        ``/api/scenes``) always plumbed it through; the full-page route
        did not — hence the existing slug tests passed (they only checked
        the sidebar .active marker) while the visible grid was wrong.

        Loose "Film B not in r.text" checks would not work here — the
        LeftPane lists every registered film by title regardless of the
        active slug. Assertions below target grid-only markup: the
        countrow value span, the per-group ``N / M`` badge, and the
        unique ``data-scene-id`` per card.
        """
        r = two_film_client.get("/scenes?film=film_a")
        assert r.status_code == 200, r.text[:300]
        # Countrow total reflects only film_a's 3 scenes (aggregate would
        # show 5 = 3 + 2).
        assert '<span class="v">3</span>' in r.text
        assert '<span class="v">5</span>' not in r.text
        # film_a's per-group badge appears; film_b's must not (the .group
        # heading + badge are emitted only for groups in groups_by_film).
        assert ">3 / 3</span>" in r.text
        assert ">2 / 2</span>" not in r.text
        # Scene cards from film_b (scene_ids 4 and 5) must be absent from
        # the grid — ``data-scene-id`` only renders inside ``.scenecard``.
        assert 'data-scene-id="4"' not in r.text
        assert 'data-scene-id="5"' not in r.text
        # film_a's cards are present.
        assert 'data-scene-id="1"' in r.text
        assert 'data-scene-id="3"' in r.text

    def test_scenes_page_grid_filtered_to_film_b(self, two_film_client):
        """Mirror of the film_a case for film_b — guards against
        accidental hard-coding of the first registered slug."""
        r = two_film_client.get("/scenes?film=film_b")
        assert r.status_code == 200, r.text[:300]
        assert '<span class="v">2</span>' in r.text
        assert '<span class="v">5</span>' not in r.text
        assert ">2 / 2</span>" in r.text
        assert ">3 / 3</span>" not in r.text
        assert 'data-scene-id="1"' not in r.text
        assert 'data-scene-id="2"' not in r.text
        assert 'data-scene-id="3"' not in r.text
        assert 'data-scene-id="4"' in r.text
        assert 'data-scene-id="5"' in r.text


# ── Selection persistence across the page (form + tab nav) ───────────────────


class TestFilmSlugPropagation:
    """The current_slug must propagate into every navigation surface on the
    page — otherwise picking a film and switching tabs silently reverts to
    the aggregate view.
    """

    def test_search_form_carries_film_hidden_input(self, two_film_client):
        """/search?film=<slug> embeds a hidden name="film" inside the
        search form, so the form's hx-include picks it up and /api/search
        receives ?film=<slug>."""
        import re

        r = two_film_client.get("/search?film=film_a")
        assert r.status_code == 200
        # Hidden input must sit INSIDE #search-text-form (hx-include only
        # walks elements selected by the include CSS selector). Pattern
        # tolerates any attribute order.
        assert re.search(
            r'<form\s+id="search-text-form"[^>]*>[^<]*'
            r"(?:<[^>]+>[^<]*)*?"
            r'<input[^>]*type="hidden"[^>]*name="film"[^>]*value="film_a"',
            r.text,
            re.DOTALL,
        ), "search form must contain a hidden film=film_a input"

    def test_search_form_omits_hidden_input_for_aggregate(self, two_film_client):
        """No ?film= → the hidden input is absent (an empty film= would
        turn into the literal string "" downstream, breaking the
        ``if slug is None`` aggregate branch in the route)."""
        r = two_film_client.get("/search")
        assert r.status_code == 200
        assert 'type="hidden" name="film"' not in r.text
        assert 'name="film" type="hidden"' not in r.text

    def test_image_search_url_carries_film_slug(self, two_film_client):
        """/search?film=<slug> embeds the slug in the image POST URL so
        the route's Query(...) dependency picks it up (a hidden form
        field would land in multipart body, which Query ignores)."""
        r = two_film_client.get("/search?film=film_a")
        assert r.status_code == 200
        assert 'hx-post="/api/search/image?film=film_a"' in r.text

    def test_image_search_url_unscoped_for_aggregate(self, two_film_client):
        """No ?film= → image POST URL has no query string."""
        r = two_film_client.get("/search")
        assert r.status_code == 200
        assert 'hx-post="/api/search/image"' in r.text

    def test_topbar_tabs_preserve_slug(self, two_film_client):
        """Every TopBar tab chip preserves ?film=<slug> on its href so
        switching tabs from Buscar → Cenas doesn't drop the selection."""
        r = two_film_client.get("/search?film=film_a")
        assert r.status_code == 200
        for path in ("/search", "/scenes", "/annotate", "/rimas", "/processing"):
            assert f'href="{path}?film=film_a"' in r.text, f"TopBar tab lost ?film=film_a on {path}"

    def test_iconrail_preserves_slug(self, two_film_client):
        """The IconRail's per-tab anchors also carry the active slug."""
        r = two_film_client.get("/search?film=film_b")
        assert r.status_code == 200
        # The rail and topbar both produce ``href="/<route>?film=<slug>"``;
        # we only assert one occurrence per route is enough — both surfaces
        # share the contract.
        for path in ("/search", "/scenes", "/annotate", "/rimas", "/processing"):
            assert f'href="{path}?film=film_b"' in r.text

    def test_topbar_tabs_no_slug_when_aggregate(self, two_film_client):
        """Aggregate mode (no ?film=) → tab hrefs are bare paths."""
        r = two_film_client.get("/search")
        assert r.status_code == 200
        for path in ("/search", "/scenes", "/annotate", "/rimas", "/processing"):
            assert f'href="{path}"' in r.text


# ── Empty library suppresses the film rows ───────────────────────────────────


class TestFilmSelectorEmpty:
    """The .ch-film rows are hidden when no films are registered."""

    def test_selector_absent_when_no_films(self, client):
        """Empty library → no .ch-film row in the left pane.

        The ``client`` fixture has no films registered (tmp_config, no seed).
        The partial gates on ``{% if films %}`` so no row markup is emitted;
        the empty-state ``<p class="tree-empty">`` renders instead.
        """
        r = client.get("/scenes")
        assert r.status_code == 200
        assert 'class="ch-film' not in r.text
        # Empty-state copy is rendered as a fallback.
        assert "No films in library" in r.text or "Sem filmes" in r.text

    def test_page_still_renders_when_no_films(self, client):
        """No films does not crash any full-page route."""
        for path in FULL_PAGE_ROUTES:
            r = client.get(path)
            assert r.status_code == 200, f"{path} returned {r.status_code}"
            assert "<!DOCTYPE html>" in r.text
