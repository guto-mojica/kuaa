"""Unit tests for the shared golden-snapshot helper (F4)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests._snapshot import SNAPSHOT_DIR, assert_snapshot, normalize


def test_normalize_rounds_floats_and_sorts_keys():
    raw = {"b": 1, "a": [0.1234567, {"z": 2, "y": 0.9999994}]}
    out = normalize(raw)
    # Floats rounded to the harness tolerance (6 dp); dict keys ordered.
    assert list(out.keys()) == ["a", "b"]
    assert out["a"][0] == 0.123457
    assert out["a"][1] == {"y": 0.999999, "z": 2}  # z>y but keys re-sorted


def test_assert_snapshot_records_then_matches(monkeypatch, tmp_path):
    monkeypatch.setattr("tests._snapshot.SNAPSHOT_DIR", tmp_path)
    monkeypatch.setenv("UPDATE_SNAPSHOTS", "1")
    # Record: writes the file and returns without asserting.
    assert_snapshot("demo_case", {"ids": [3, 1, 2], "score": 0.50000001})
    written = json.loads((tmp_path / "demo_case.json").read_text())
    assert written == {"ids": [3, 1, 2], "score": 0.5}
    # Compare: same value passes.
    monkeypatch.delenv("UPDATE_SNAPSHOTS")
    assert_snapshot("demo_case", {"ids": [3, 1, 2], "score": 0.5})


def test_assert_snapshot_diffs_on_drift(monkeypatch, tmp_path):
    monkeypatch.setattr("tests._snapshot.SNAPSHOT_DIR", tmp_path)
    (tmp_path / "drift.json").write_text(json.dumps({"ids": [1, 2]}))
    with pytest.raises(AssertionError) as ei:
        assert_snapshot("drift", {"ids": [2, 1]})
    assert "drift" in str(ei.value)


def test_assert_snapshot_missing_without_update(monkeypatch, tmp_path):
    monkeypatch.setattr("tests._snapshot.SNAPSHOT_DIR", tmp_path)
    monkeypatch.delenv("UPDATE_SNAPSHOTS", raising=False)
    with pytest.raises(AssertionError) as ei:
        assert_snapshot("never_recorded", {"x": 1})
    assert "UPDATE_SNAPSHOTS=1" in str(ei.value)
