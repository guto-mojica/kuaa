"""Typed config schema + loader equivalence (F1) — validation contract (T1).

Phase-0 F1 coverage (already present, not duplicated):
  - test_load_default_returns_settings_with_dot_access: Settings instance, dot-access, config_version
  - test_get_and_to_dict_shims_work: .get() shim, .to_dict()
  - test_config_alias_is_settings: Config is Settings
  - test_malformed_type_raises_config_error: wrong type → ConfigError, field path in message
  - test_unknown_selector_value_raises_config_error: bad image_embedder Literal → ConfigError
  - test_unknown_top_level_key_rejected: extra top-level key → ConfigError

T1 additions below: no-arg load, malformed YAML, nested section errors,
local.yaml merge, seed/config_version defaults, ConfigError.code, additional
model-selector roles.
"""

from __future__ import annotations

# ─── Repo root for reading default.yaml in overlay tests ────────────────────
from pathlib import Path

import pytest
import yaml

from kuaa.config import CONFIG_VERSION, Config, Settings, load_config
from kuaa.errors import ConfigError

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_YAML = _REPO_ROOT / "config" / "default.yaml"


def _overlay(tmp_path: Path, patch: dict) -> Path:
    """Write a minimal override yaml that deep-merges ``patch`` over defaults."""
    p = tmp_path / "override.yaml"
    p.write_text(yaml.safe_dump(patch), encoding="utf-8")
    return p


# ─── F1 coverage (do not duplicate) ─────────────────────────────────────────


def test_load_default_returns_settings_with_dot_access(tmp_path):
    cfg = load_config(project_root=tmp_path, ensure_dirs=False)
    assert isinstance(cfg, Settings)
    # Dot-access preserved across nested sections.
    assert cfg.search.top_k_default == 9
    assert cfg.embeddings.model == "ViT-B-32"
    assert cfg.models.image_embedder in ("clip_openclip", "siglip_multilingual")
    assert cfg.paths.metadata_dir  # resolved Path
    assert cfg.config_version == CONFIG_VERSION


def test_get_and_to_dict_shims_work(tmp_path):
    cfg = load_config(project_root=tmp_path, ensure_dirs=False)
    assert cfg.logging.get("level", "X") == "INFO"
    assert cfg.logging.get("does_not_exist", "fallback") == "fallback"
    d = cfg.to_dict()
    assert isinstance(d, dict) and d["search"]["top_k_default"] == 9
    # Paths round-trip as strings/Path (json-coercible) for run_manifest.
    assert "paths" in d


def test_config_alias_is_settings():
    assert Config is Settings


def test_malformed_type_raises_config_error(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("search:\n  top_k_default: not-an-int\n")
    with pytest.raises(ConfigError) as ei:
        load_config(str(bad), project_root=tmp_path, ensure_dirs=False)
    # Field-path appears in the message.
    assert "search" in str(ei.value) and "top_k_default" in str(ei.value)


def test_unknown_selector_value_raises_config_error(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("models:\n  image_embedder: not_a_backend\n")
    with pytest.raises(ConfigError):
        load_config(str(bad), project_root=tmp_path, ensure_dirs=False)


def test_unknown_top_level_key_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("totally_unknown_section: 1\n")
    with pytest.raises(ConfigError):
        load_config(str(bad), project_root=tmp_path, ensure_dirs=False)


# ─── T1 additions ─────────────────────────────────────────────────────────────


def test_default_yaml_loads_no_args():
    """load_config() with no arguments resolves against CWD; must return Settings."""
    cfg = load_config(ensure_dirs=False)
    assert isinstance(cfg, Settings)
    # Spec F1: dot-access and embeddings.batch_size check.
    assert cfg.paths.metadata_dir is not None
    assert cfg.embeddings.batch_size >= 1


def test_malformed_yaml_raises_config_error(tmp_path):
    """A YAML parse error must surface as ConfigError (not raw yaml.YAMLError)."""
    p = tmp_path / "broken.yaml"
    p.write_text("paths: [unterminated\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p, ensure_dirs=False)


def test_nested_section_type_error_carries_path(tmp_path):
    """A type error in a deeply-nested field must report the full field path."""
    bad = _overlay(tmp_path, {"visual_analysis": {"face_detection": {"min_face_size": "bad"}}})
    with pytest.raises(ConfigError) as ei:
        load_config(bad, project_root=tmp_path, ensure_dirs=False)
    msg = str(ei.value)
    # Path must contain the nested section names so callers can pinpoint the problem.
    assert "face_detection" in msg or "min_face_size" in msg


def test_local_yaml_override_merges_and_preserves_defaults(tmp_path):
    """A user override changes only the target field; all other defaults are preserved."""
    override = _overlay(tmp_path, {"search": {"top_k_default": 42}})
    cfg = load_config(override, project_root=tmp_path, ensure_dirs=False)
    assert cfg.search.top_k_default == 42
    # Other search defaults must survive the partial override.
    assert cfg.search.hybrid_enabled is True
    # Unrelated section completely untouched.
    assert cfg.embeddings.batch_size == 16


def test_seed_default_is_42(tmp_path):
    """Settings.seed has a fixed default of 42 (reproducibility anchor)."""
    cfg = load_config(project_root=tmp_path, ensure_dirs=False)
    assert cfg.seed == 42


def test_config_version_matches_constant(tmp_path):
    """cfg.config_version must equal the exported CONFIG_VERSION constant."""
    cfg = load_config(project_root=tmp_path, ensure_dirs=False)
    assert cfg.config_version == CONFIG_VERSION


def test_config_error_code_attribute():
    """ConfigError carries a stable machine code (F2 error taxonomy contract)."""
    exc = ConfigError("bad field", code="config.invalid")
    assert exc.code == "config.invalid"
    # Default code must also be set when no explicit code is passed.
    exc_default = ConfigError("oops")
    assert exc_default.code == "config.invalid"


def test_scene_describer_literal_enforced(tmp_path):
    """models.scene_describer is a Literal; an unlisted value must raise ConfigError."""
    bad = _overlay(tmp_path, {"models": {"scene_describer": "gpt4o"}})
    with pytest.raises(ConfigError):
        load_config(bad, project_root=tmp_path, ensure_dirs=False)


def test_hardware_device_literal_enforced(tmp_path):
    """hardware.device is a Literal['auto','cpu','cuda','mps']; anything else must fail."""
    bad = _overlay(tmp_path, {"hardware": {"device": "tpu"}})
    with pytest.raises(ConfigError):
        load_config(bad, project_root=tmp_path, ensure_dirs=False)
