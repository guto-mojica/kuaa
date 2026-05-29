"""Shared golden-snapshot helper (F4).

One ergonomic record/compare/update primitive every behavior-preserving
refactor uses to prove it changed nothing. Record or refresh with
``UPDATE_SNAPSHOTS=1 uv run pytest ...``; without the flag a missing or
drifted snapshot fails loudly with a unified diff.

Float tolerance: values are rounded to ``FLOAT_NDIGITS`` decimal places
before compare so CPU/GPU/backend score noise (~1e-7) does not flap.
"""
from __future__ import annotations

import difflib
import json
import os
from pathlib import Path
from typing import Any

SNAPSHOT_DIR: Path = Path(__file__).parent / "fixtures" / "snapshots"
FLOAT_NDIGITS = 6


def normalize(value: object) -> Any:
    """Return a JSON-stable, float-tolerant projection of ``value``.

    Dicts get sorted keys; floats are rounded to :data:`FLOAT_NDIGITS`;
    lists/tuples recurse (order preserved — ranking order is significant).
    """
    if isinstance(value, float):
        return round(value, FLOAT_NDIGITS)
    if isinstance(value, dict):
        return {k: normalize(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [normalize(v) for v in value]
    return value


def _dumps(value: object) -> str:
    return json.dumps(normalize(value), indent=2, ensure_ascii=False, sort_keys=False) + "\n"


def assert_snapshot(name: str, value: object) -> None:
    """Compare ``value`` against the recorded snapshot ``name``.

    With ``UPDATE_SNAPSHOTS=1`` set, (re)writes the snapshot and returns.
    Otherwise asserts equality, raising ``AssertionError`` with a diff on
    drift or a record hint when the snapshot file is absent.
    """
    path = SNAPSHOT_DIR / f"{name}.json"
    rendered = _dumps(value)
    if os.environ.get("UPDATE_SNAPSHOTS") == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
        return
    if not path.exists():
        raise AssertionError(
            f"Snapshot {name!r} missing at {path}. "
            f"Record it with UPDATE_SNAPSHOTS=1 uv run pytest ..."
        )
    expected = path.read_text(encoding="utf-8")
    if rendered != expected:
        diff = "".join(
            difflib.unified_diff(
                expected.splitlines(keepends=True),
                rendered.splitlines(keepends=True),
                fromfile=f"{name} (recorded)",
                tofile=f"{name} (current)",
            )
        )
        raise AssertionError(f"Snapshot drift for {name!r}:\n{diff}")
