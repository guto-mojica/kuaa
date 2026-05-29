"""Typed config schema + loader equivalence (F1)."""

from __future__ import annotations

import pytest

from cinemateca.config import CONFIG_VERSION, Config, Settings, load_config
from cinemateca.errors import ConfigError


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
