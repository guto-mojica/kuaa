"""CLI surface coverage for audio_extract / audio_embed."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from cinemateca.__main__ import _resolve_steps, app


def test_resolve_steps_accepts_audio_names():
    out = _resolve_steps("audio_extract,audio_embed")
    assert out == {"audio_extract", "audio_embed"}


def test_resolve_steps_accepts_mixed_with_legacy_aliases():
    out = _resolve_steps("scenes,audio_extract,llm")
    assert out == {"scene_detection", "audio_extract", "llm_description"}


def test_resolve_steps_rejects_unknown():
    import typer

    with pytest.raises(typer.BadParameter):
        _resolve_steps("audio_extract,nope")


def test_library_reembed_help_lists_audio_steps():
    runner = CliRunner()
    res = runner.invoke(app, ["library", "reembed", "--help"])
    assert res.exit_code == 0
    assert "audio_extract" in res.output
    assert "audio_embed" in res.output
