"""2.2: ``enrich_rhyme`` loads each film's keyframe metadata once per request.

The Rimas grid build shares one ``kf_cache`` across every echo. A run of echoes
from the same film must parse ``keyframes_metadata.json`` once, not twice per
echo (URL + timecode). The resolved url/timecode must stay identical to the
legacy two-resolver path (first-row-wins per scene).
"""

from __future__ import annotations

import kuaa.rhymes.enrich as enrich_mod
from kuaa.rhymes.algorithm import Rhyme


def test_enrich_shares_keyframe_index_across_echoes(monkeypatch, tmp_path):
    calls: list[str] = []
    fp = tmp_path / "porter" / "frames" / "x-Scene-007-01.jpg"
    by_scene = {7: {"filepath": str(fp), "start_time_s": 12.0}}

    def fake_index(cfg, slug):
        calls.append(slug)
        return by_scene, 24.0, tmp_path

    monkeypatch.setattr(enrich_mod, "keyframe_index", fake_index)

    rhymes = [Rhyme("porter", 7, 0.9), Rhyme("porter", 7, 0.8), Rhyme("porter", 7, 0.7)]
    cache: dict = {}
    out = [enrich_mod.enrich_rhyme(None, r, {}, kf_cache=cache) for r in rhymes]

    assert calls == ["porter"]  # loaded once for 3 echoes, not 3x
    assert out[0]["keyframe_url"] == "/media/porter/frames/x-Scene-007-01.jpg"
    assert out[0]["timecode"] == "00:00:12:00"  # to_smpte(12.0, 24)
    assert out[0]["film_title"] == "porter"


def test_enrich_without_cache_loads_each_time(monkeypatch, tmp_path):
    calls: list[str] = []

    def fake_index(cfg, slug):
        calls.append(slug)
        return {}, 24.0, tmp_path

    monkeypatch.setattr(enrich_mod, "keyframe_index", fake_index)
    rhymes = [Rhyme("porter", 7, 0.9), Rhyme("porter", 7, 0.8)]
    out = [enrich_mod.enrich_rhyme(None, r, {}) for r in rhymes]

    assert calls == ["porter", "porter"]  # no cache → per-echo load (legacy behaviour)
    # Unresolvable scene (empty index) collapses to empty url/timecode.
    assert out[0]["keyframe_url"] == ""
    assert out[0]["timecode"] == ""
