"""Multi-film aggregate context / search tests."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from api.services.catalog import build_scenes_context_aggregate
from cinemateca.library import register_film


def _make_film(library_dir: Path, slug: str, scene_ids: list[int]) -> None:
    """Create a minimal per-film layout with N scenes + a tag index entry."""
    md = library_dir / slug / "metadata"
    md.mkdir(parents=True)
    (library_dir / slug / "frames" / "keyframes").mkdir(parents=True)
    (library_dir / slug / "raw").mkdir()
    (library_dir / slug / "raw" / f"{slug}.mp4").write_bytes(b"")
    kf_meta = [
        {
            "scene_id": sid,
            "filepath": f"data/library/{slug}/frames/keyframes/{sid}.jpg",
            "start_time_s": float(sid),
        }
        for sid in scene_ids
    ]
    (md / "keyframes_metadata.json").write_text(json.dumps(kf_meta))
    (md / "scene_tags.json").write_text(
        json.dumps({"outdoor": scene_ids})
    )


def _cfg(library_dir: Path) -> object:
    return SimpleNamespace(paths=SimpleNamespace(library_dir=str(library_dir)))


def test_aggregate_scenes_context_merges_films(tmp_path: Path) -> None:
    library_dir = tmp_path / "library"
    library_dir.mkdir()
    register_film(library_dir, slug="a", title="A", year=2000, raw_filename="a.mp4")
    register_film(library_dir, slug="b", title="B", year=2001, raw_filename="b.mp4")
    _make_film(library_dir, "a", [0, 1, 2])
    _make_film(library_dir, "b", [0, 1])

    ctx = build_scenes_context_aggregate(_cfg(library_dir))

    assert len(ctx["cards"]) == 5
    assert {c["film_slug"] for c in ctx["cards"]} == {"a", "b"}
    assert "outdoor" in ctx["available_tags"]
    assert ctx["no_data"] is False


def test_aggregate_scenes_context_empty_when_no_films(tmp_path: Path) -> None:
    library_dir = tmp_path / "library"
    library_dir.mkdir()
    ctx = build_scenes_context_aggregate(_cfg(library_dir))
    assert ctx["cards"] == []
    assert ctx["no_data"] is True
