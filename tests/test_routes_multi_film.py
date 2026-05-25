"""End-to-end smoke tests for ?film=<slug> multi-film route support.

Task T9 — Routes accept ?film=<slug>.

These tests build a two-film library layout under the isolated
``tmp_config`` (shared conftest fixture), register both films, seed
per-film metadata, then hit the routes with and without ?film=<slug>
to assert:

  1. No-slug → aggregate view includes content from both films.
  2. ?film=<slug> → only that film's content is returned.
  3. /api/search?q=...&film=<slug> → results filtered to that film.

All tests run without a real CLIP model (no index seeded) so the
search assertions focus on the no-index graceful-degradation path.
Search-with-real-index coverage lives in test_multi_film_search.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# ── Two-film fixture helpers ──────────────────────────────────────────────────


def _seed_film(library_dir: Path, slug: str, title: str, scene_ids: list[int]) -> None:
    """Register a film and create its per-film directory layout.

    Creates ``library_dir/<slug>/{raw,metadata,frames/keyframes,embeddings}/``.
    Writes ``keyframes_metadata.json`` with *scene_ids* scenes and
    ``scene_tags.json`` with a single ``"outdoor"`` tag on all scenes.
    """
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
def two_film_client(tmp_config, monkeypatch, client):
    """``TestClient`` with a two-film library: film_a (3 scenes) + film_b (2 scenes).

    Builds on the ``tmp_config`` + ``client`` isolation pair. Populates
    ``library_dir`` with films "film_a" (title "Film A") and "film_b"
    (title "Film B") and their per-film metadata, so aggregate-view routes
    return content from both films.
    """
    library_dir = Path(tmp_config.paths.library_dir)
    _seed_film(library_dir, slug="film_a", title="Film A", scene_ids=[1, 2, 3])
    _seed_film(library_dir, slug="film_b", title="Film B", scene_ids=[4, 5])
    return client


# ── Scenes aggregate / per-film ──────────────────────────────────────────────


class TestScenesRouteMultiFilm:
    """``/tab/scenes`` and ``/api/scenes`` dispatch on ``?film=``.

    Task 15 (Mojica) swapped the legacy ``<p>N scenes</p>`` count line
    for the ``.countrow`` markup with the total in a ``<span class="v">``
    span followed by the localised ``scenes`` label. These tests pin
    the same scene-count contract on the new structure: the count
    appears as the value-span content and the per-film group's
    ``match_count`` displays as ``<span class="ct">N / M</span>``.
    """

    def test_tab_scenes_aggregate_includes_both_films(self, two_film_client):
        r = two_film_client.get("/tab/scenes")
        assert r.status_code == 200, r.text[:300]
        # Both films have scenes → no-data state must be absent.
        assert "No scenes found" not in r.text
        # New countrow: total scenes in the value-span. 5 = 3 (film_a) + 2 (film_b).
        assert '<span class="v">5</span>' in r.text
        # Both per-film group headings appear (one each for film_a / film_b).
        assert "Film A" in r.text
        assert "Film B" in r.text

    def test_tab_scenes_per_film_a_filters(self, two_film_client):
        r = two_film_client.get("/tab/scenes?film=film_a")
        assert r.status_code == 200, r.text[:300]
        # Only film_a's 3 scenes are present in the countrow total.
        assert '<span class="v">3</span>' in r.text
        assert "Film A" in r.text
        # film_b's heading must not appear in the per-film view.
        assert "Film B" not in r.text

    def test_tab_scenes_per_film_b_filters(self, two_film_client):
        r = two_film_client.get("/tab/scenes?film=film_b")
        assert r.status_code == 200, r.text[:300]
        assert '<span class="v">2</span>' in r.text
        assert "Film B" in r.text
        assert "Film A" not in r.text

    def test_tab_scenes_unknown_slug_falls_back_to_aggregate(self, two_film_client):
        """Unknown slug → film_slug_query returns None → aggregate view (200).

        film_slug_query validates the slug against the library dir and
        silently returns None for unregistered slugs rather than raising,
        so the tab renders the aggregate grid instead of 500-ing.
        """
        r = two_film_client.get("/tab/scenes?film=ghost")
        assert r.status_code == 200


class TestScenesGridAggregate:
    """/api/scenes (no ?film=) uses the aggregate grid path.

    Task 15 (Mojica): the grid partial renders ``.group`` headings +
    ``.scenecard`` articles; the legacy ``<p>N scenes</p>`` count line
    is gone. The countrow lives in ``scenes.html`` (the toolbar parent
    template); the grid fragment only renders the per-group
    ``<span class="ct">N / M</span>`` badges + the cards themselves.
    These tests pin the count via the group badge + the explicit
    ``data-scene-id`` attributes that uniquely identify each card.
    """

    def test_aggregate_grid_returns_all_scenes(self, two_film_client):
        """No slug → all 5 scenes from both films are returned."""
        r = two_film_client.get("/api/scenes")
        assert r.status_code == 200, r.text[:300]
        # film_a has 3 scenes, film_b has 2 → both groups appear with
        # their respective match counts (``<span class="ct">N / M</span>``).
        assert "Film A" in r.text
        assert "Film B" in r.text
        assert ">3 / 3</span>" in r.text
        assert ">2 / 2</span>" in r.text
        # Cards from both films appear: scene IDs 1-3 from film_a (scene_id
        # 1, 2, 3) and 4-5 from film_b → check several scene-card ids.
        assert 'data-scene-id="1"' in r.text
        assert 'data-scene-id="4"' in r.text

    def test_per_film_grid_filters_to_film_a(self, two_film_client):
        r = two_film_client.get("/api/scenes", params={"film": "film_a"})
        assert r.status_code == 200, r.text[:300]
        assert "Film A" in r.text
        assert ">3 / 3</span>" in r.text
        # scene_ids 4 and 5 belong to film_b and must be absent.
        assert 'data-scene-id="4"' not in r.text
        assert 'data-scene-id="5"' not in r.text

    def test_per_film_grid_filters_to_film_b(self, two_film_client):
        r = two_film_client.get("/api/scenes", params={"film": "film_b"})
        assert r.status_code == 200, r.text[:300]
        assert "Film B" in r.text
        assert ">2 / 2</span>" in r.text
        # scene_ids 1-3 belong to film_a and must be absent.
        assert 'data-scene-id="1"' not in r.text

    def test_aggregate_grid_with_tag_filter(self, two_film_client):
        """Tag filter in aggregate mode: both films seeded with 'outdoor' on
        all scenes, so ?tags=outdoor should return all 5 scenes."""
        r = two_film_client.get("/api/scenes", params={"tags": "outdoor"})
        assert r.status_code == 200, r.text[:300]
        # outdoor tag matches all scenes in both films → both groups still
        # appear at full match count.
        assert ">3 / 3</span>" in r.text
        assert ">2 / 2</span>" in r.text

    def test_current_slug_no_crash_on_aggregate(self, two_film_client):
        """Aggregate /api/scenes (no film param) must return 200 without crash.

        current_slug=None is injected into context but the template does
        not render it yet (T10 adds sidebar highlighting). Key contract:
        no 500.
        """
        r = two_film_client.get("/api/scenes")
        assert r.status_code == 200, r.text[:300]

    def test_current_slug_no_crash_on_per_film(self, two_film_client):
        """Per-film /api/scenes injects current_slug — no crash."""
        r = two_film_client.get("/api/scenes", params={"film": "film_a"})
        assert r.status_code == 200, r.text[:300]


# ── Library filter ────────────────────────────────────────────────────────────


class TestLibraryFilterMultiFilm:
    """``/api/library/filter`` lists all films; ``?q=`` narrows."""

    def test_filter_lists_both_films(self, two_film_client):
        r = two_film_client.get("/api/library/filter")
        assert r.status_code == 200, r.text[:300]
        assert "Film A" in r.text
        assert "Film B" in r.text

    def test_filter_query_narrows_to_one_film(self, two_film_client):
        r = two_film_client.get("/api/library/filter", params={"q": "Film A"})
        assert r.status_code == 200, r.text[:300]
        assert "Film A" in r.text
        assert "Film B" not in r.text


# ── Search no-index graceful degradation ─────────────────────────────────────


class TestSearchRouteMultiFilm:
    """``/api/search`` dispatches on ``?film=``.

    No CLIP index is seeded, so both the aggregate and per-film paths
    return the no-index graceful-degradation response (not a crash).
    Full search-with-real-index coverage is in test_multi_film_search.py.
    """

    def test_api_search_aggregate_no_index_degrades_gracefully(self, two_film_client):
        r = two_film_client.get("/api/search", params={"q": "outdoor"})
        assert r.status_code == 200, r.text[:300]
        # No index → the no-index hint is rendered, no 500.
        assert "No search index found" in r.text

    def test_api_search_per_film_no_index_degrades_gracefully(self, two_film_client):
        r = two_film_client.get("/api/search", params={"q": "outdoor", "film": "film_a"})
        assert r.status_code == 200, r.text[:300]
        assert "No search index found" in r.text
