"""One keyframe and one description per scene, on every screen.

A scene owns several keyframe rows and several raw description records. The
default keyframe is the middle row and the default description is the
canonical one; only the Search card -> right pane hand-off carries an explicit
keyframe name.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from api.services.scenes import build_inspector_context
from kuaa.annotations.descriptions import canonical_description
from kuaa.library.keyframes import pick_keyframe_entry
from kuaa.rhymes.metadata import load_scene_meta
from kuaa.scene_ids import scene_id_key
from kuaa.search._lookup import enrich_hits_with_film_metadata

SID = 351
NAMES = ["s351_kf_01.jpg", "s351_kf_02.jpg", "s351_kf_03.jpg"]
TEXTS = [
    "A harbour at first light.",
    "Two men argue on the quay.",
    "A ship leaves the dock slowly.",
]


@pytest.fixture()
def seeded(seed_metadata, tmp_config):
    # keyframe_url only resolves files that sit under data_dir.
    frames = Path(tmp_config.paths.data_dir) / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    scenes = []
    for i, name in enumerate(NAMES):
        (frames / name).touch()
        scenes.append(
            {
                "scene_id": SID,
                "filepath": str(frames / name),
                "start_time_s": 83.0 + i,
                "end_time_s": 90.0,
            }
        )
    scenes.append(
        {
            "scene_id": 352,
            "filepath": str(frames / "s352_kf_01.jpg"),
            "start_time_s": 120.0,
            "end_time_s": 128.0,
        }
    )
    descs = [
        {"scene_id": SID, "description": t, "keyframe": n}
        for t, n in zip(TEXTS, NAMES, strict=True)
    ]
    seed_metadata(scenes=scenes, descriptions=descs)
    return tmp_config, scenes, descs


def test_pick_keyframe_entry_rules(seeded):
    _, scenes, _ = seeded
    assert pick_keyframe_entry(scenes, SID) is scenes[1]
    assert pick_keyframe_entry(scenes, SID, NAMES[2]) is scenes[2]
    assert pick_keyframe_entry(scenes, SID, "s352_kf_01.jpg") is scenes[1]
    assert pick_keyframe_entry(scenes, SID, "../../etc/passwd") is scenes[1]
    assert pick_keyframe_entry(scenes, 999) is None


def test_inspector_context_honours_kf(seeded):
    cfg, _, _ = seeded
    ctx = build_inspector_context(cfg, scene_id=SID, slug="default", kf=NAMES[2])
    assert ctx is not None
    assert ctx["selected_scene"]["keyframe_url"].endswith(NAMES[2])
    assert ctx["selected_scene"]["kf"] == NAMES[2]
    ctx = build_inspector_context(cfg, scene_id=SID, slug="default")
    assert ctx["selected_scene"]["keyframe_url"].endswith(NAMES[1])
    assert ctx["selected_scene"]["kf"] == NAMES[1]


def test_inspector_route_carries_kf_through_tabs(seeded, client):
    r = client.get(f"/api/scenes/{SID}/inspector?film=default&kf={NAMES[2]}")
    assert r.status_code == 200
    assert NAMES[2] in r.text
    for tab in ("activity", "annotations", "properties"):
        needle = f"tab={tab}"
        hrefs = [ln for ln in r.text.splitlines() if needle in ln and "hx-get" in ln]
        assert hrefs, tab
        assert all(f"kf={NAMES[2]}" in h for h in hrefs), tab


def test_rhymes_anchor_uses_middle_row_and_canonical_description(seeded):
    cfg, _, descs = seeded
    anchor = load_scene_meta(cfg, "default", SID)
    assert anchor is not None
    assert anchor["keyframe_url"].endswith(NAMES[1])
    expected = canonical_description(descs)[scene_id_key(SID)]["description"]
    assert anchor["description"] == expected


def test_search_card_description_matches_inspector(seeded):
    cfg, _, _ = seeded
    (card,) = enrich_hits_with_film_metadata(cfg, [{"film_slug": "default", "scene_id": SID}])
    ctx = build_inspector_context(cfg, scene_id=SID, slug="default")
    assert card["description"] == ctx["selected_scene"]["description"]
