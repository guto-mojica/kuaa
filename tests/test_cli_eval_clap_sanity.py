"""CLI surface for ``cinemateca eval clap-sanity``.

The command runs a canned slate of CLAP archival-audio queries and asserts
each one's P@5 clears the recorded floor (default 0.4). Tests stub out the
heavy bits (config load, FilmContext, embedder, audio index, search) so the
suite stays fast and hermetic — the live retrieval gate runs separately as
the pre-commit sanity check the M3 plan documents.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner


@pytest.fixture
def fixture_path(tmp_path: Path) -> Path:
    """Synthetic clap_sanity_queries.json with one query, three expected ids."""
    p = tmp_path / "clap_sanity_queries.json"
    p.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "library": "jeca_tatu",
                "p_at_5_floor": 0.4,
                "queries": [
                    {
                        "id": "q1",
                        "query": "music",
                        "expected_scene_ids": [10, 20, 30],
                    },
                ],
            }
        )
    )
    return p


def _patch_environment(monkeypatch: pytest.MonkeyPatch, *, hits: list[dict]) -> None:
    """Stub the four collaborators the clap-sanity command resolves at runtime.

    - ``load_config`` returns a bare object (the CLI never inspects it; only
      ``FilmContext.for_film`` and ``get_audio_embedder`` do, both stubbed).
    - ``FilmContext.for_film`` returns a stub with a ``metadata_dir`` attribute
      (the CLI builds ``audio_dir = metadata_dir.parent / "audio"``).
    - ``load_audio_index`` returns a sentinel (the stubbed ``search_audio``
      ignores it).
    - ``get_audio_embedder`` returns a sentinel.
    - ``search_audio`` returns the caller-provided hits.
    """
    from types import SimpleNamespace

    monkeypatch.setattr(
        "cinemateca.config.load_config",
        lambda *a, **k: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "cinemateca.library.context.FilmContext.for_film",
        classmethod(
            lambda cls, cfg, slug: SimpleNamespace(
                slug=slug, metadata_dir=Path("/tmp/fake-film/metadata")
            )
        ),
    )
    monkeypatch.setattr(
        "cinemateca.search.audio.load_audio_index",
        lambda audio_dir: object(),
    )
    monkeypatch.setattr(
        "cinemateca.models.registry.get_audio_embedder",
        lambda cfg, device=None: object(),
    )
    monkeypatch.setattr(
        "cinemateca.search.audio.search_audio",
        lambda idx, emb, q, top_k: hits,
    )


def test_clap_sanity_emits_per_query_metrics(
    monkeypatch: pytest.MonkeyPatch, fixture_path: Path
) -> None:
    """3 of 3 expected scenes in top-5 → P@5 = 0.6, above the 0.4 floor → PASS."""
    from cinemateca.__main__ import app

    _patch_environment(
        monkeypatch,
        hits=[
            {"scene_id": 10, "score": 0.9},
            {"scene_id": 20, "score": 0.8},
            {"scene_id": 99, "score": 0.7},
            {"scene_id": 30, "score": 0.6},
            {"scene_id": 88, "score": 0.5},
        ],
    )
    runner = CliRunner()
    result = runner.invoke(app, ["eval", "clap-sanity", "--fixture", str(fixture_path)])
    assert result.exit_code == 0, result.stdout
    assert "P@5" in result.stdout
    assert "PASS" in result.stdout
    assert "0.60" in result.stdout  # 3/5 expected scenes hit


def test_clap_sanity_fails_when_below_floor(
    monkeypatch: pytest.MonkeyPatch, fixture_path: Path
) -> None:
    """0 of 3 expected scenes in top-5 → P@5 = 0.0, below floor → FAIL + exit 1."""
    from cinemateca.__main__ import app

    _patch_environment(
        monkeypatch,
        hits=[{"scene_id": 999 + i, "score": 0.1} for i in range(5)],
    )
    runner = CliRunner()
    result = runner.invoke(app, ["eval", "clap-sanity", "--fixture", str(fixture_path)])
    assert result.exit_code == 1
    assert "FAIL" in result.stdout


def test_clap_sanity_missing_index_fails_fast(
    monkeypatch: pytest.MonkeyPatch, fixture_path: Path
) -> None:
    """When ``load_audio_index`` returns ``None`` the CLI must exit 1 with a clear
    FAIL message — the only path to a missing-index state on a real install."""
    from types import SimpleNamespace

    from cinemateca.__main__ import app

    monkeypatch.setattr("cinemateca.config.load_config", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(
        "cinemateca.library.context.FilmContext.for_film",
        classmethod(
            lambda cls, cfg, slug: SimpleNamespace(
                slug=slug, metadata_dir=Path("/tmp/fake-film/metadata")
            )
        ),
    )
    monkeypatch.setattr("cinemateca.search.audio.load_audio_index", lambda audio_dir: None)
    runner = CliRunner()
    result = runner.invoke(app, ["eval", "clap-sanity", "--fixture", str(fixture_path)])
    assert result.exit_code == 1
    assert "FAIL" in result.stdout
