"""Tests for the eval seed module + CLI (Task 33).

The seed module ships a small bundle of sample queries used to populate
the /eval grading UI on a fresh install. We verify three things:

1. ``write_seed()`` writes a well-formed JSON file shaped exactly the way
   ``api.services.eval_service._load_queries`` expects (list of dicts
   with ``id``, ``text``, ``results``; each result with ``scene_id``,
   ``film_slug``, ``score``).
2. The bundled ``SAMPLE_QUERIES`` constant is internally complete — no
   silent missing fields that would crash the rows template.
3. ``write_seed()`` creates intermediate directories on demand (the CLI
   passes ``data/eval`` which may not exist on a fresh checkout).
4. ``count`` is clamped so the CLI degrades gracefully instead of
   raising ``IndexError`` on ``--queries 99``.
"""

from __future__ import annotations

import json
from pathlib import Path

from cinemateca.eval.seed import SAMPLE_QUERIES, write_seed


def test_write_seed_creates_well_formed_file(tmp_path: Path):
    """The output file matches the contract _load_queries reads."""

    out = write_seed(tmp_path, "test-run", count=3)
    assert out.exists()
    assert out.name == "test-run.queries.json"

    data = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) == 3

    # Per-query contract (Task 30 + queue/rows templates).
    first = data[0]
    for key in ("id", "text", "source", "lang", "k", "candidate_count", "results"):
        assert key in first, f"missing {key} on seeded query"
    assert isinstance(first["results"], list)
    assert len(first["results"]) == 9

    # Per-result contract (rows.html reads these directly).
    first_result = first["results"][0]
    for key in (
        "scene_id",
        "film_slug",
        "film_title",
        "year",
        "timecode",
        "description",
        "tags",
        "score",
        "keyframe_url",
    ):
        assert key in first_result, f"missing {key} on candidate result"


def test_sample_queries_all_complete():
    """Every bundled query carries the fields the templates depend on."""

    assert len(SAMPLE_QUERIES) >= 5
    for q in SAMPLE_QUERIES:
        assert "id" in q and isinstance(q["id"], int)
        assert "text" in q and isinstance(q["text"], str) and q["text"]
        assert "results" in q and len(q["results"]) > 0
        for r in q["results"]:
            assert "scene_id" in r and isinstance(r["scene_id"], int)
            assert "film_slug" in r and isinstance(r["film_slug"], str)
            assert "score" in r and isinstance(r["score"], (int, float))
            # Keyframe URL is built off scene_id + film_slug — verify
            # the helper produced the canonical /media/library/... path
            # the FastAPI static mount serves.
            assert r["keyframe_url"].startswith("/media/library/")


def test_write_seed_creates_nested_root_dir(tmp_path: Path):
    """The CLI passes a root that may not exist; write_seed mkdirs it."""

    nested = tmp_path / "deep" / "nested" / "eval"
    assert not nested.exists()
    out = write_seed(nested, "deep", count=1)
    assert out.exists()
    assert out.parent == nested


def test_write_seed_clamps_count_above_max(tmp_path: Path):
    """Passing --queries 99 shouldn't crash; cap at the bundled max."""

    out = write_seed(tmp_path, "clamped", count=99)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data) == len(SAMPLE_QUERIES)


def test_write_seed_handles_zero_count(tmp_path: Path):
    """--queries 0 writes an empty list (still valid JSON the loader accepts)."""

    out = write_seed(tmp_path, "empty", count=0)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data == []
