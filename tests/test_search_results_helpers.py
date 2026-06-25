"""Unit tests for kuaa.search._results + kuaa.search._lookup.

The plan-mandated tests for T8. ``TestResultsToDicts`` in
``tests/test_search_service.py`` and
``test_build_search_context_aggregate_unions_and_filters`` in
``tests/test_multi_film_search.py`` continue to exercise the full
behaviour through ``api.services.search``'s re-exports; this file
pins the relocated symbols at their new home so a future renamer can't
silently break the import path.
"""

from __future__ import annotations

import pandas as pd

from kuaa.search._lookup import mojica_search_defaults
from kuaa.search._results import results_to_dicts


def test_results_to_dicts_without_meta(tmp_path):
    df = pd.DataFrame(
        [
            {
                "scene_id": 1,
                "similarity": 0.9,
                "filepath": str(tmp_path / "1.jpg"),
                "rank": 1,
            }
        ]
    )
    out = results_to_dicts(df, tmp_path)
    assert len(out) == 1
    assert "img_url" in out[0]
    assert "timecode" not in out[0]


def test_results_to_dicts_with_meta_adds_timecode(tmp_path):
    df = pd.DataFrame(
        [
            {
                "scene_id": 1,
                "similarity": 0.9,
                "filepath": str(tmp_path / "1.jpg"),
                "rank": 1,
            }
        ]
    )
    meta = {1: {"start_time_s": 60.0}}
    out = results_to_dicts(df, tmp_path, meta_by_scene=meta, fps=24.0)
    assert out[0]["timecode"] == "00:01:00:00"


def test_mojica_defaults_shape():
    d = mojica_search_defaults()
    assert d["query"] == ""
    assert d["active_mode"] == "text"
    assert d["results"] == []
    assert d["highlighted_tags"] == set()
