"""Tests for the command palette (Phase 7 / Task 27).

Covers:

  * ``GET /api/palette/search`` with no query — returns every static
    catalogue row + every registered film.
  * Substring filtering by label.
  * The palette scaffold is present on every full-page tab (so ⌘K works
    from anywhere).
  * The client JS bundle is served by FastAPI's StaticFiles mount.
"""

from __future__ import annotations


def test_palette_search_no_query_returns_all_groups(client):
    """Empty ``q`` returns all four groups, with navigate/actions populated."""
    r = client.get("/api/palette/search")
    assert r.status_code == 200
    data = r.json()
    # Stable JSON shape: every group key is present even if empty.
    for key in ("navigate", "actions", "films", "scenes_recent"):
        assert key in data, f"missing group {key} in response"
    # The static catalogues match palette_service.NAVIGATE / ACTIONS.
    assert len(data["navigate"]) == 6
    assert len(data["actions"]) == 3
    # Films may be empty in the hermetic test client (no registered films).
    assert isinstance(data["films"], list)
    assert isinstance(data["scenes_recent"], list)


def test_palette_search_filters_by_label(client):
    """A substring query keeps matching rows and drops non-matching ones."""
    r = client.get("/api/palette/search", params={"q": "annot"})
    assert r.status_code == 200
    data = r.json()
    # ``Annotate`` is in navigate; the action labels do NOT contain "annot".
    nav_labels = [i["label"].lower() for i in data["navigate"]]
    assert any("annot" in lbl for lbl in nav_labels)
    # ``Home`` should not survive the filter.
    assert not any("home" in lbl for lbl in nav_labels)
    # Shape is preserved.
    for key in ("navigate", "actions", "films", "scenes_recent"):
        assert key in data


def test_palette_search_case_insensitive(client):
    """The query is folded to lower-case before substring match."""
    r_lower = client.get("/api/palette/search", params={"q": "search"})
    r_upper = client.get("/api/palette/search", params={"q": "SEARCH"})
    assert r_lower.status_code == 200
    assert r_upper.status_code == 200
    assert r_lower.json() == r_upper.json()


def test_palette_root_present_on_every_tab(client):
    """The ``#palette`` scaffold is rendered server-side on every page.

    palette.js targets ``getElementById('palette')`` at IIFE evaluation,
    so the scaffold must be in the initial HTML for ⌘K to work without a
    full-page reload after navigation.
    """
    for path in ("/", "/search", "/scenes", "/annotate", "/rimas", "/processing"):
        r = client.get(path)
        assert r.status_code == 200, f"{path} returned {r.status_code}"
        assert 'id="palette"' in r.text, f"{path} missing #palette scaffold"
        # Sanity-check the input + list anchors palette.js binds to.
        assert 'id="cp-input"' in r.text, f"{path} missing #cp-input"
        assert 'id="cp-list"' in r.text, f"{path} missing #cp-list"


def test_palette_js_is_served(client):
    """The static palette.js bundle is reachable and exports ``Palette``."""
    r = client.get("/static/js/palette.js")
    assert r.status_code == 200
    body = r.text
    assert "window.Palette" in body
    assert "cp-input" in body


def test_palette_search_returns_registered_films(client, seed_metadata):
    """Films registered in ``library_dir`` appear in the ``films`` group."""
    seed_metadata()  # registers slug "default" / title "Default Film"
    r = client.get("/api/palette/search")
    assert r.status_code == 200
    films = r.json()["films"]
    assert any(f["slug"] == "default" for f in films)
    default_film = next(f for f in films if f["slug"] == "default")
    assert default_film["label"] == "Default Film"
    assert default_film["url"] == "/scenes?film=default"
    assert default_film["icon"] == "film"


def test_palette_search_returns_registered_scenes(client, seed_metadata):
    """Processed scenes appear in the palette scene group and deep-link back."""
    seed_metadata()
    r = client.get("/api/palette/search", params={"q": "walking"})
    assert r.status_code == 200
    scenes = r.json()["scenes_recent"]
    assert any(s["scene_id"] == 351 for s in scenes)
    scene = next(s for s in scenes if s["scene_id"] == 351)
    assert scene["url"] == "/scenes?film=default&scene=351"
    assert scene["badge"] == "scene"


def test_palette_search_film_label_filter(client, seed_metadata):
    """Film-label substring matches scope ``films`` correctly."""
    seed_metadata()
    # "default" matches the seed film's title "Default Film".
    r = client.get("/api/palette/search", params={"q": "default"})
    data = r.json()
    assert any(f["slug"] == "default" for f in data["films"])
    # A query that doesn't match the title should drop the film.
    r2 = client.get("/api/palette/search", params={"q": "zzzzzz"})
    assert r2.json()["films"] == []
