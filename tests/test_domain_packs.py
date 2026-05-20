from __future__ import annotations

import pytest

from cinemateca.config import load_config
from cinemateca.domain import (
    DomainError,
    export_record,
    load_domain_from_config,
    load_domain_pack,
    prompt_dict,
    resolve_domain_pack_path,
)
from cinemateca.models.describer._common import PROMPTS
from cinemateca.eval.datasets import load_dataset


def test_archive_domain_loads_and_matches_current_prompt_keys():
    pack = load_domain_pack("config/domains/archive.yaml")

    assert pack.id == "archive"
    assert pack.label == "Film archive"
    assert set(prompt_dict(pack)) == set(PROMPTS)
    assert prompt_dict(pack)["description"] == PROMPTS["description"]
    assert "description" in pack.export_mapping


def test_default_config_selects_archive_domain():
    cfg = load_config(project_root=".")

    path = resolve_domain_pack_path(cfg, project_root=".")
    pack = load_domain_from_config(cfg, project_root=".")

    assert path.as_posix().endswith("config/domains/archive.yaml")
    assert pack.id == "archive"


def test_media_broadcast_domain_changes_prompt_and_export_shape():
    pack = load_domain_pack("config/domains/media_broadcast.yaml")
    prompts = prompt_dict(pack)

    assert pack.id == "media_broadcast"
    assert "shot_type" in prompts
    assert "licensing_notes" in prompts
    assert set(prompts) != set(PROMPTS)

    row = {
        "scene_id": 101,
        "keyframe_path": "frames/101.jpg",
        "description": "Reporter standup",
        "objects": ["microphone"],
        "tags": ["reporter"],
        "_raw_responses": {
            "shot_type": "medium",
            "visible_people": "1 reporter",
            "location_type": "outdoor",
            "action": "speaking to camera",
            "logos_or_text": "microphone flag",
            "licensing_notes": "recognizable face",
            "reusable_broll_score": "3",
        },
    }

    exported = export_record(row, pack)
    assert exported["shot_type"] == "medium"
    assert exported["licensing_notes"] == "recognizable face"
    assert "location" not in exported


def test_media_broadcast_eval_queries_follow_m2_schema():
    dataset = load_dataset("data/eval/media_broadcast_queries.yaml")

    assert dataset.dataset == "media_broadcast_sample"
    assert len(dataset.queries) >= 5


def test_invalid_domain_pack_reports_missing_required_fields(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        """
id: bad
label: Bad Pack
metadata_fields: []
prompt_templates: {}
export_mapping: {}
""",
        encoding="utf-8",
    )

    with pytest.raises(DomainError, match="metadata_fields"):
        load_domain_pack(bad)


def test_explicit_domain_path_overrides_pack_name(tmp_path):
    custom = tmp_path / "custom.yaml"
    custom.write_text(
        """
id: custom
label: Custom
metadata_fields:
  - name: description
    label: Description
    type: text
prompt_templates:
  description:
    prompt: Describe the scene.
    max_new_tokens: 12
export_mapping:
  description: description
""",
        encoding="utf-8",
    )
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        f"""
domain:
  pack: missing
  path: {custom}
""",
        encoding="utf-8",
    )
    cfg = load_config(cfg_path, project_root=tmp_path)

    assert load_domain_from_config(cfg, project_root=tmp_path).id == "custom"
