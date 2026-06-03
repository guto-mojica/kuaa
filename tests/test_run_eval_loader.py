"""#2: run_eval's loader auto-detects the multimodal m3_full shape.

``--all-modes`` (and the text path) must use ``_load_text_dataset``, which
extracts the text subset from an m3_full file. The strict ``load_dataset``
rejects m3_full image rows that legitimately carry no ``relevant_scene_ids``
(the bug the reviewer reproduced: ``image-01.relevant_scene_ids must contain
at least one id``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

M3_FULL = _REPO / "data" / "eval" / "m3_full_queries.yaml"


def test_load_text_dataset_handles_multimodal_m3_full() -> None:
    from run_eval import _load_text_dataset

    if not M3_FULL.exists():
        pytest.skip(f"fixture missing: {M3_FULL}")
    ds = _load_text_dataset(M3_FULL)
    # The text subset loaded without raising on the image rows.
    assert len(ds.queries) >= 1


def test_strict_load_dataset_rejects_multimodal_m3_full() -> None:
    """Documents the bug: the legacy strict loader chokes on m3_full image rows.

    This is why ``_all_modes`` must route through ``_load_text_dataset`` rather
    than calling ``load_dataset`` directly.
    """
    from cinemateca.eval.datasets import load_dataset

    with pytest.raises(Exception):  # noqa: B017,PT011 — any load error documents the regression
        load_dataset(M3_FULL)
