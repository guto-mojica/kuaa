"""run_eval's loader auto-detects the multimodal query-set shape.

``--all-modes`` (and the text path) must use ``_load_text_dataset``, which
extracts the text subset. The strict ``load_dataset`` rejects image rows that
legitimately carry no ``relevant_scene_ids``: ``image-01.relevant_scene_ids
must contain at least one id``.

The fixture is written inline rather than read from ``data/eval/``. The file
this test used to point at was retired with the pre-corpus01 query sets, and a
strict-loader assertion that passes because its input is missing documents
nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))


@pytest.fixture
def multimodal_yaml(tmp_path: Path) -> Path:
    """A query set carrying one text row with labels and one unlabelled image row.

    The image_path must exist on disk — load_modal_queries validates it — so it
    points at a file written into tmp_path.
    """
    img = tmp_path / "anchor.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg")
    yaml_path = tmp_path / "queries.yaml"
    yaml_path.write_text(
        f"""\
dataset: loader_test
version: 1
queries:
  - id: text-01
    query_type: text
    text: "crane lifting a crate onto a ship"
    lang: en
    relevant_scene_ids: [6]
    relevance:
      6: 2
    notes: "labelled text row"
  - id: image-01
    query_type: image
    text: "(image query) anchor frame"
    image_path: "{img}"
    lang: en
    notes: "unlabelled image row — the one the strict loader rejects"
""",
        encoding="utf-8",
    )
    return yaml_path


def test_load_text_dataset_handles_multimodal(multimodal_yaml: Path) -> None:
    from run_eval import _load_text_dataset

    ds = _load_text_dataset(multimodal_yaml)
    # The text subset loaded without raising on the image row.
    assert [q.id for q in ds.queries] == ["text-01"]


def test_strict_load_dataset_rejects_multimodal(multimodal_yaml: Path) -> None:
    """Documents the bug: the strict loader chokes on unlabelled image rows.

    This is why ``_all_modes`` must route through ``_load_text_dataset`` rather
    than calling ``load_dataset`` directly.
    """
    from kuaa.eval.datasets import load_dataset

    with pytest.raises(Exception):  # noqa: B017,PT011 — any load error documents the regression
        load_dataset(multimodal_yaml)
