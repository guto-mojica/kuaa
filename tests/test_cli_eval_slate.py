"""CLI surface for ``cinemateca eval slate`` (E3b).

The command loads an ``m3_full``-shaped query YAML, calls the real
per-modality slate generator (``cinemateca.eval.slates.generate_slate``)
for every query of the requested modality, and writes a
``<root>/<run_id>.queries.json`` file in the SAME rows-template contract
``cinemateca.eval.seed.write_seed`` produces — so the ``/eval`` grading
page renders generated slates exactly as it renders the seeded ones.

This test is hermetic: ``generate_slate`` is monkeypatched (by the name
as bound in ``eval_cmd``) with a 2-row stub, so no real index, model, or
disk artefact is touched. The assertion mirrors ``tests/test_eval_seed.py``
— the written file is a list of query dicts, each carrying ``id`` / ``text``
/ ``results``, and each result carrying all nine rows-template keys.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

# The nine keys every candidate row must carry so rows.html renders.
_ROWS_KEYS = {
    "scene_id",
    "film_slug",
    "film_title",
    "year",
    "timecode",
    "description",
    "tags",
    "score",
    "keyframe_url",
}


def _two_row_stub(*, query, cfg, library_dir, k=9) -> list[dict[str, Any]]:
    """Stand-in for ``generate_slate``: two valid 9-key candidate rows."""
    return [
        {
            "scene_id": 12,
            "film_slug": "jeca_tatu",
            "film_title": "Jeca Tatu",
            "year": 1959,
            "timecode": "00:00:30",
            "description": "stub row one",
            "tags": ["a", "b"],
            "score": 0.91,
            "keyframe_url": "/media/library/jeca_tatu/frames/scene_0012.jpg",
        },
        {
            "scene_id": 34,
            "film_slug": "jeca_tatu",
            "film_title": "Jeca Tatu",
            "year": 1959,
            "timecode": "00:01:10",
            "description": "stub row two",
            "tags": [],
            "score": 0.42,
            "keyframe_url": "/media/library/jeca_tatu/frames/scene_0034.jpg",
        },
    ]


@pytest.fixture
def one_image_yaml(tmp_path: Path) -> Path:
    """A minimal m3_full-shaped YAML with a single image query.

    The image_path must exist on disk (load_modal_queries validates it),
    so we point it at a fixture file we write into tmp_path.
    """
    img = tmp_path / "anchor.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg")
    yaml_path = tmp_path / "q.yaml"
    yaml_path.write_text(
        f"""\
dataset: m3_test
version: 1
queries:
  - id: image-01
    query_type: image
    text: "(image query) anchor frame"
    image_path: "{img}"
    lang: en
    notes: "stub image query"
""",
        encoding="utf-8",
    )
    return yaml_path


def test_eval_slate_writes_rows_contract(
    monkeypatch: pytest.MonkeyPatch, one_image_yaml: Path, tmp_path: Path
) -> None:
    """``eval slate --modality image`` writes <run>.queries.json in the rows contract."""
    import cinemateca.commands.eval_cmd as eval_cmd
    from cinemateca.__main__ import app

    # Patch the name as bound in eval_cmd so no real backend runs.
    monkeypatch.setattr(eval_cmd, "generate_slate", _two_row_stub)

    root = tmp_path / "eval_out"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "eval",
            "slate",
            "--modality",
            "image",
            "--queries",
            str(one_image_yaml),
            "--run",
            "t",
            "--root",
            str(root),
        ],
    )
    assert result.exit_code == 0, result.stdout

    out_path = root / "t.queries.json"
    assert out_path.exists(), f"expected {out_path} written"

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) == 1

    first = data[0]
    # Per-query contract (mirrors test_eval_seed.py).
    for key in ("id", "text", "source", "lang", "k", "candidate_count", "results"):
        assert key in first, f"missing {key} on generated query"
    assert first["id"] == "image-01"
    assert isinstance(first["results"], list)
    assert len(first["results"]) == 2

    # Per-result contract — all nine rows-template keys.
    for r in first["results"]:
        assert set(r.keys()) == _ROWS_KEYS, f"row key drift: {sorted(r)}"
