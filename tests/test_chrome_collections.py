"""Left-pane collection counts match the Scenes grid and carry correct labels."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from api.services.chrome_service import build_chrome_context
from api.services.scenes import build_cenas_context
from tests.test_routes_multi_film import _seed_film

# scene_id -> (tags, description); three keyframe rows per scene are written.
SCENES: dict[int, tuple[list[str], str]] = {
    1: (["interior"], "A kitchen."),
    2: (["exterior"], "A field."),
    3: (["exterior"], "Title card with white letters."),
    4: (["duas-pessoas"], "Two people talk."),
    5: ([], "Nothing notable."),
    6: (["baixa-luz"], "A dim room."),
}
EXPECTED = {"exterior": 1, "cartela": 1, "dialogo": 1, "interior": 2}


@pytest.fixture()
def library(tmp_config):
    lib = Path(tmp_config.paths.library_dir)
    _seed_film(lib, slug="film_a", title="Film A", scene_ids=list(SCENES))
    meta = lib / "film_a" / "metadata"
    rows = [
        {
            "scene_id": sid,
            "filepath": f"library/film_a/frames/keyframes/{sid}_{k}.jpg",
            "start_time_s": float(sid * 60 + k),
            "end_time_s": float(sid * 60 + 10),
        }
        for sid in SCENES
        for k in range(3)
    ]
    (meta / "keyframes_metadata.json").write_text(json.dumps(rows))
    tags: dict[str, list[int]] = {}
    for sid, (scene_tags, _) in SCENES.items():
        for t in scene_tags:
            tags.setdefault(t, []).append(sid)
    (meta / "scene_tags.json").write_text(json.dumps(tags))
    descs = [{"scene_id": sid, "description": d} for sid, (_, d) in SCENES.items()]
    (meta / "scene_descriptions.json").write_text(json.dumps(descs))
    return tmp_config


def _by_category(cfg) -> dict[str | None, dict]:
    cols = build_chrome_context(cfg)["collections"]
    return {c["category"]: c for c in cols}


def _grid_count(cfg, bucket: str) -> int:
    ctx = build_cenas_context(cfg, bucket=bucket, limit=1000)
    return sum(len(g["scenes"]) for g in ctx["groups_by_film"])


def test_bucket_counts_equal_grid_and_expected(library) -> None:
    cols = _by_category(library)
    for bucket, expected in EXPECTED.items():
        assert cols[bucket]["count"] == _grid_count(library, bucket) == expected, bucket


def test_buckets_do_not_exceed_entire_library(library) -> None:
    cols = _by_category(library)
    total = sum(cols[b]["count"] for b in EXPECTED)
    assert total <= cols[None]["count"]


def test_interior_entry_is_labelled_interiors(library) -> None:
    entry = next(
        c
        for c in build_chrome_context(library)["collections"]
        if c["url"] == "/scenes?bucket=interior"
    )
    assert entry["label"] == "Interiors"


def test_left_pane_translates_collections_to_pt(library, raw_client) -> None:
    raw_client.cookies.set("locale", "pt_BR")
    html = raw_client.get("/scenes").text
    assert "Todo o acervo" in html
    assert "Interiores" in html
    assert "Entire library" not in re.sub(r"<!--.*?-->", "", html, flags=re.S)
    assert "Night scenes" not in html
