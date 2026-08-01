"""The shared canonical-description loader used by every retrieval path.

``scene_descriptions.json`` holds N caption records per scene (one per
keyframe). Each consumer used to pick its own — BM25 kept the last row, the
reranker kept the first, the Scenes tab used the canonical middle — so a scene
could be indexed on, reranked on, and displayed as three different captions.
"""

from __future__ import annotations

import json
from pathlib import Path

from kuaa.annotations.descriptions import load_canonical_descriptions


def _write(md: Path, records: list[dict]) -> None:
    md.mkdir(parents=True, exist_ok=True)
    (md / "scene_descriptions.json").write_text(json.dumps(records))


def _scene(sid: int, kf: int, text: str) -> dict:
    return {"scene_id": sid, "keyframe_id": f"scene_{sid:04d}_kf_{kf:02d}", "description": text}


def test_picks_the_positional_middle_record(tmp_path: Path) -> None:
    _write(
        tmp_path,
        [
            _scene(1, 1, "First caption."),
            _scene(1, 2, "Middle caption."),
            _scene(1, 3, "Last caption."),
        ],
    )
    [record] = load_canonical_descriptions(tmp_path)
    assert record["keyframe_id"] == "scene_0001_kf_02"
    assert record["description"].startswith("Middle caption.")


def test_middle_pick_is_independent_of_file_order(tmp_path: Path) -> None:
    """Rows are ordered by keyframe id, not by however the describer resumed."""
    _write(
        tmp_path,
        [_scene(1, 3, "Last."), _scene(1, 1, "First."), _scene(1, 2, "Middle.")],
    )
    [record] = load_canonical_descriptions(tmp_path)
    assert record["keyframe_id"] == "scene_0001_kf_02"


def test_folds_novel_sibling_sentences_into_the_canonical_text(tmp_path: Path) -> None:
    """Every sibling caption already cost an LLM inference; keep their content."""
    _write(
        tmp_path,
        [
            _scene(1, 1, "A man walks a dog."),
            _scene(1, 2, "A man stands in a yard."),
            _scene(1, 3, "A car passes behind him."),
        ],
    )
    [record] = load_canonical_descriptions(tmp_path)
    assert "A man stands in a yard." in record["description"]
    assert "A car passes behind him." in record["description"]
    assert "walks a dog" in record["description"]


def test_returns_one_record_per_scene(tmp_path: Path) -> None:
    _write(
        tmp_path,
        [_scene(1, 1, "One."), _scene(1, 2, "Two."), _scene(2, 1, "Three."), _scene(2, 2, "Four.")],
    )
    records = load_canonical_descriptions(tmp_path)
    assert sorted(r["scene_id"] for r in records) == [1, 2]


def test_rebuilds_after_the_file_changes(tmp_path: Path) -> None:
    """The loader is cached on (mtime_ns, size); a curator edit must win."""
    _write(tmp_path, [_scene(1, 1, "Original text.")])
    assert load_canonical_descriptions(tmp_path)[0]["description"] == "Original text."

    _write(tmp_path, [_scene(1, 1, "Edited text entirely.")])
    assert load_canonical_descriptions(tmp_path)[0]["description"] == "Edited text entirely."


def test_callers_cannot_mutate_each_others_records(tmp_path: Path) -> None:
    _write(tmp_path, [_scene(1, 1, "Shared.")])
    first = load_canonical_descriptions(tmp_path)
    first[0]["description"] = "clobbered"
    assert load_canonical_descriptions(tmp_path)[0]["description"] == "Shared."


def test_missing_or_malformed_file_yields_no_records(tmp_path: Path) -> None:
    assert load_canonical_descriptions(tmp_path / "absent") == []
    (tmp_path / "scene_descriptions.json").write_text(json.dumps({"not": "a list"}))
    assert load_canonical_descriptions(tmp_path) == []
