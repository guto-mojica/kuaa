"""Route-level coverage for ``api/routes/library.py``.

Focused on the management endpoints that have no other test surface
(``/api/library/remove/{slug}``, ``/api/library/add``). The
``/api/library/filter`` and ``/api/library/tree`` filter routes are
covered indirectly by ``test_routes_multi_film.py``.
"""

from __future__ import annotations

import json
from pathlib import Path


def test_remove_film_route_passes_slug_as_kwarg(tmp_config, client) -> None:
    """``POST /api/library/remove/{slug}`` must call ``delete_film`` with
    the keyword-only ``slug=`` argument.

    Regression: the route previously called ``delete_film(library_dir, slug)``
    positionally, but the signature is ``delete_film(library_dir: Path, *,
    slug: str)``. A positional call raises ``TypeError`` at runtime — the
    server returned 500 and the film was never removed from the registry.
    """
    from kuaa.library import load_registry, register_film

    library_dir = Path(tmp_config.paths.library_dir)
    register_film(
        library_dir,
        slug="ghost_film",
        title="Ghost Film",
        year=None,
        raw_filename="ghost_film.mp4",
    )
    assert "ghost_film" in load_registry(library_dir)

    response = client.post("/api/library/remove/ghost_film")

    assert response.status_code == 200, response.text
    assert "ghost_film" not in load_registry(library_dir)


def test_remove_film_route_wipe_deletes_film_dir(tmp_config, client) -> None:
    """``?wipe=`` non-empty triggers ``shutil.rmtree`` on the per-film dir."""
    from kuaa.library import register_film

    library_dir = Path(tmp_config.paths.library_dir)
    register_film(
        library_dir,
        slug="wipe_me",
        title="Wipe Me",
        year=None,
        raw_filename="wipe_me.mp4",
    )
    film_dir = library_dir / "wipe_me"
    film_dir.mkdir(exist_ok=True)
    (film_dir / "raw").mkdir(exist_ok=True)
    (film_dir / "raw" / "wipe_me.mp4").touch()
    (film_dir / "metadata").mkdir(exist_ok=True)
    (film_dir / "metadata" / "keyframes_metadata.json").write_text(json.dumps([]))

    response = client.post("/api/library/remove/wipe_me", data={"wipe": "1"})

    assert response.status_code == 200, response.text
    assert not film_dir.exists()


def test_remove_film_route_unknown_slug_is_idempotent(tmp_config, client) -> None:
    """Removing a slug that's not in the registry is a no-op (200, no raise).

    The route checks ``slug in registry`` before calling ``delete_film``, so
    a stale form submission (slug already removed) returns the tree HTML
    without surfacing a 500.
    """
    response = client.post("/api/library/remove/nonexistent_slug")

    assert response.status_code == 200, response.text
