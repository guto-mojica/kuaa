"""The right-pane manual-tags form must edit manual tags only.

The save route replaces a scene's manual tags with the field's contents, so a
field prefilled from the merged list would copy every automatic tag into
``manual_annotations.json``.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

import pytest

from api.services.scenes import build_inspector_context

SID = 351
KF = "s351_kf_01.jpg"
AUTO = ["exterior", "dia"]
MANUAL = "hand-made"


@pytest.fixture()
def seeded(seed_metadata, tmp_config):
    frames = Path(tmp_config.paths.data_dir) / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    (frames / KF).touch()
    scenes = [
        {
            "scene_id": SID,
            "filepath": str(frames / KF),
            "start_time_s": 83.0,
            "end_time_s": 90.0,
        }
    ]
    seed_metadata(
        scenes=scenes,
        llm_tags={"exterior": [SID], "dia": [SID]},
        manual={str(SID): [MANUAL]},
    )
    return tmp_config


def _field_value(page: str) -> str:
    m = re.search(r'<input[^>]*name="tags"[^>]*value="([^"]*)"', page, re.S)
    assert m, "tags input not found"
    return html.unescape(m.group(1))


def _manual_file(cfg) -> dict:
    path = Path(cfg.paths.library_dir) / "default" / "metadata" / "manual_annotations.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_context_separates_manual_tags(seeded):
    ctx = build_inspector_context(seeded, scene_id=SID, slug="default")
    scene = ctx["selected_scene"]
    assert scene["manual_tags"] == [MANUAL]
    assert set(AUTO) <= set(scene["tags"])


def test_form_prefilled_with_manual_tags_only(seeded, client):
    r = client.get(f"/api/scenes/{SID}/inspector?tab=annotations&film=default")
    assert r.status_code == 200
    value = _field_value(r.text)
    assert MANUAL in value
    assert not any(t in value for t in AUTO)


def test_save_round_trip_keeps_manual_tags_only(seeded, client):
    r = client.get(f"/api/scenes/{SID}/inspector?tab=annotations&film=default")
    value = _field_value(r.text)
    saved = client.post(
        "/api/annotate/save?film=default",
        data={"scene_id": SID, "filter": "all", "tags": value},
    )
    assert saved.status_code == 200
    assert _manual_file(seeded)[str(SID)] == [MANUAL]


def test_form_swaps_nothing_and_refetch_keeps_kf(seeded, client):
    r = client.get(f"/api/scenes/{SID}/inspector?tab=annotations&film=default&kf={KF}")
    form = re.search(r"<form[^>]*>", r.text, re.S).group(0)
    assert 'hx-swap="none"' in form
    assert f"kf={KF}" in form
