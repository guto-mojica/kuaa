"""A7: limit/offset pagination on the scene grid + search results."""

from __future__ import annotations


def test_scenes_grid_respects_limit(seed_metadata, client) -> None:
    # Seed a dataset with more scenes than the page size, assert the grid pages.
    seed_metadata(
        scenes=[
            {
                "scene_id": i,
                "filepath": f"f{i}.jpg",
                "start_time_s": float(i),
                "end_time_s": float(i) + 1.0,
            }
            for i in range(1, 61)
        ]
    )
    r1 = client.get("/api/scenes?limit=10&offset=0")
    r2 = client.get("/api/scenes?limit=10&offset=10")
    assert r1.status_code == 200 and r2.status_code == 200
    # Distinct pages -> distinct HTML (scene ids differ).
    assert r1.text != r2.text


def test_scenes_limit_is_capped(client) -> None:
    # limit above the cap is rejected (422) by the Pydantic constraint.
    r = client.get("/api/scenes?limit=9999")
    assert r.status_code == 422


def test_search_offset_param_accepted(client) -> None:
    r = client.get("/api/search?q=river&retriever=clip&top_k=5&offset=5")
    assert r.status_code == 200
