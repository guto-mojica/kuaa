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
    """``/tab/scenes`` and ``/api/scenes`` dispatch on ``?film=``."""

    def test_tab_scenes_aggregate_includes_both_films(self, two_film_client):
        r = two_film_client.get("/tab/scenes")
        assert r.status_code == 200, r.text[:300]
        # Both films have scenes → no-data state must be absent.
        assert "No scenes found" not in r.text
        # The scenes grid shows cards from both films.
        assert "5 scenes" in r.text

    def test_tab_scenes_per_film_a_filters(self, two_film_client):
        r = two_film_client.get("/tab/scenes?film=film_a")
        assert r.status_code == 200, r.text[:300]
        assert "3 scenes" in r.text

    def test_tab_scenes_per_film_b_filters(self, two_film_client):
        r = two_film_client.get("/tab/scenes?film=film_b")
        assert r.status_code == 200, r.text[:300]
        assert "2 scenes" in r.text

    def test_tab_scenes_unknown_slug_raises(self, two_film_client):
        """Unknown slug → FilmContext.for_film raises ValueError.

        The route does not currently catch ValueError from for_film (that
        guard lands in T10). TestClient propagates server exceptions by
        default; the test uses pytest.raises to pin this behavior — an
        exception is visible, a silent wrong result is not.
        """
        with pytest.raises(ValueError, match="No such film directory"):
            two_film_client.get("/tab/scenes?film=ghost")


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

    def test_api_search_aggregate_no_index_degrades_gracefully(
        self, two_film_client
    ):
        r = two_film_client.get("/api/search", params={"q": "outdoor"})
        assert r.status_code == 200, r.text[:300]
        # No index → the no-index hint is rendered, no 500.
        assert "No search index found" in r.text

    def test_api_search_per_film_no_index_degrades_gracefully(
        self, two_film_client
    ):
        r = two_film_client.get("/api/search", params={"q": "outdoor", "film": "film_a"})
        assert r.status_code == 200, r.text[:300]
        assert "No search index found" in r.text
