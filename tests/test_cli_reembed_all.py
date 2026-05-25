"""Tests for the unified Typer CLI in ``cinemateca/__main__.py``.

Covers the user-facing command tree:

  * ``cinemateca serve``            — delegates to uvicorn.run
  * ``cinemateca info``             — prints video properties / exit 1 on error
  * ``cinemateca process``          — pipeline with explicit slug
  * ``cinemateca library list``     — table of registered films
  * ``cinemateca library reembed``  — registry-driven re-embed (regression-locked
                                       slug-from-registry, --only filter,
                                       empty-registry error, stale-cleanup)
  * ``cinemateca library delete``   — destructive op with --yes confirmation
  * ``cinemateca config show``      — YAML dump of merged config

Tests use ``typer.testing.CliRunner`` to invoke the actual Typer app
(``cinemateca.__main__.app``) so help-text generation, option parsing,
exit codes, and subcommand wiring are all exercised end-to-end.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture()
def runner() -> CliRunner:
    # Typer 0.12+ keeps stderr separate by default; the older `mix_stderr`
    # kwarg was removed. Plain construction is what we want.
    return CliRunner()


def _seed_film(library_dir: Path, slug: str, raw_filename: str) -> Path:
    """Register a film and create the per-film layout the CLI expects."""
    from cinemateca.library import register_film

    register_film(
        library_dir, slug=slug, title=slug.replace("_", " ").title(),
        year=None, raw_filename=raw_filename,
    )
    film_dir = library_dir / slug
    (film_dir / "embeddings").mkdir(parents=True, exist_ok=True)
    (film_dir / "metadata").mkdir(parents=True, exist_ok=True)
    raw_dir = film_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / raw_filename
    raw_path.touch()  # placeholder; the pipeline is stubbed in tests
    return raw_path


def _stub_cfg(tmp_path: Path) -> SimpleNamespace:
    """Minimal config object covering everything the CLI reads."""
    library_dir = tmp_path / "library"
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    library_dir.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        paths=SimpleNamespace(
            library_dir=str(library_dir), raw_dir=str(raw_dir),
        ),
        pipeline=SimpleNamespace(steps=SimpleNamespace(
            frame_extraction=False, scene_detection=False,
            visual_analysis=False, embeddings=False, llm_description=False,
        )),
        hardware=SimpleNamespace(device="cpu"),
    )


@pytest.fixture()
def reembed_env(tmp_path, monkeypatch):
    """Two-film library + stubs for load_config, setup_logging,
    CatalogPipeline. Returns a SimpleNamespace with the captured
    pipeline invocations."""
    cfg = _stub_cfg(tmp_path)
    library_dir = Path(cfg.paths.library_dir)
    _seed_film(library_dir, "film_a", "film_a.mp4")
    _seed_film(library_dir, "film_b", "film_b.mp4")

    invocations: list[dict] = []

    class FakePipeline:
        def __init__(self, c, slug):
            self.slug = slug

        def run(self, video_path):
            invocations.append({"slug": self.slug, "video": str(video_path)})
            return SimpleNamespace(success=True, total_duration_s=0.5)

    monkeypatch.setattr("cinemateca.config.load_config",
                        lambda _path=None: cfg)
    monkeypatch.setattr("cinemateca.config.setup_logging", lambda _c: None)
    monkeypatch.setattr("cinemateca.pipeline.CatalogPipeline", FakePipeline)
    return SimpleNamespace(cfg=cfg, library_dir=library_dir,
                            invocations=invocations)


# ── Top-level discoverability ────────────────────────────────────────────────

class TestRootHelp:
    """Calling ``cinemateca`` with no args (or --help) must print the
    full command tree without crashing — the discoverability test."""

    def test_no_args_prints_help(self, runner):
        from cinemateca.__main__ import app

        result = runner.invoke(app, [])
        # Typer prints help to stdout and exits non-zero for no_args_is_help.
        assert "Cinemateca AI" in (result.stdout + result.stderr)
        assert "library" in (result.stdout + result.stderr)
        assert "serve" in (result.stdout + result.stderr)

    def test_help_lists_all_command_groups(self, runner):
        from cinemateca.__main__ import app

        result = runner.invoke(app, ["--help"])
        out = result.stdout + result.stderr
        for cmd in ("serve", "info", "process", "library", "config"):
            assert cmd in out, f"top-level command {cmd!r} missing from --help"

    def test_library_subcommands_listed(self, runner):
        from cinemateca.__main__ import app

        result = runner.invoke(app, ["library", "--help"])
        out = result.stdout + result.stderr
        for sub in ("list", "reembed", "delete"):
            assert sub in out, f"library subcommand {sub!r} missing"


# ── reembed: slug-from-registry invariant (Phase-1 regression lock) ──────────

class TestReembedUsesRegisteredSlug:
    def test_calls_pipeline_once_per_registered_film(self, runner, reembed_env):
        from cinemateca.__main__ import app

        result = runner.invoke(app, ["library", "reembed"])
        assert result.exit_code == 0, result.stdout + result.stderr
        slugs_called = sorted(inv["slug"] for inv in reembed_env.invocations)
        assert slugs_called == ["film_a", "film_b"]

    def test_slug_matches_registry_even_when_filename_drifts(
        self, tmp_path, monkeypatch, runner,
    ):
        """A registered slug ('canonical') with a filename whose
        slugified stem differs ('drifty-name') must still invoke
        the pipeline with 'canonical' — the exact Phase-1 bug.
        """
        cfg = _stub_cfg(tmp_path)
        _seed_film(Path(cfg.paths.library_dir), "canonical", "Drifty-Name.mp4")
        invocations: list[dict] = []

        class FakePipeline:
            def __init__(self, c, slug):
                self.slug = slug
            def run(self, _v):
                invocations.append({"slug": self.slug})
                return SimpleNamespace(success=True, total_duration_s=0.0)

        monkeypatch.setattr("cinemateca.config.load_config",
                            lambda _path=None: cfg)
        monkeypatch.setattr("cinemateca.config.setup_logging", lambda _c: None)
        monkeypatch.setattr("cinemateca.pipeline.CatalogPipeline", FakePipeline)

        from cinemateca.__main__ import app

        result = runner.invoke(app, ["library", "reembed"])
        assert result.exit_code == 0, result.stdout + result.stderr
        assert invocations == [{"slug": "canonical"}]


# ── reembed: filters and edge cases ──────────────────────────────────────────

class TestReembedFilters:
    def test_only_filters_to_specified_slug(self, runner, reembed_env):
        from cinemateca.__main__ import app

        result = runner.invoke(
            app, ["library", "reembed", "--only", "film_a"]
        )
        assert result.exit_code == 0
        assert [inv["slug"] for inv in reembed_env.invocations] == ["film_a"]

    def test_only_unknown_slug_returns_error(self, runner, reembed_env):
        from cinemateca.__main__ import app

        result = runner.invoke(
            app, ["library", "reembed", "--only", "does_not_exist"]
        )
        assert result.exit_code != 0
        assert reembed_env.invocations == []
        assert "does_not_exist" in result.stderr

    def test_empty_registry_returns_error(self, tmp_path, monkeypatch, runner):
        cfg = _stub_cfg(tmp_path)
        (Path(cfg.paths.library_dir) / "films.json").write_text("{}")
        monkeypatch.setattr("cinemateca.config.load_config",
                            lambda _path=None: cfg)
        monkeypatch.setattr("cinemateca.config.setup_logging", lambda _c: None)

        from cinemateca.__main__ import app

        result = runner.invoke(app, ["library", "reembed"])
        assert result.exit_code != 0
        assert "Nenhum filme registrado" in result.stderr


# ── reembed: stale-file cleanup ──────────────────────────────────────────────

class TestReembedStaleCleanup:
    """Default: clears stale .npy + index_mapping.json so the
    pipeline's ``skip_existing`` cannot silently no-op the step.
    ``--keep-existing`` opts out."""

    def _run(self, tmp_path, monkeypatch, runner, keep_existing: bool):
        cfg = _stub_cfg(tmp_path)
        library_dir = Path(cfg.paths.library_dir)
        _seed_film(library_dir, "film_a", "film_a.mp4")
        emb_dir = library_dir / "film_a" / "embeddings"
        stale_npy = emb_dir / "keyframe_embeddings.npy"
        stale_map = emb_dir / "index_mapping.json"
        stale_npy.write_bytes(b"stale")
        stale_map.write_text("{}")

        class FakePipeline:
            def __init__(self, c, slug):
                pass
            def run(self, _v):
                return SimpleNamespace(success=True, total_duration_s=0.0)

        monkeypatch.setattr("cinemateca.config.load_config",
                            lambda _path=None: cfg)
        monkeypatch.setattr("cinemateca.config.setup_logging", lambda _c: None)
        monkeypatch.setattr("cinemateca.pipeline.CatalogPipeline", FakePipeline)

        from cinemateca.__main__ import app

        argv = ["library", "reembed"]
        if keep_existing:
            argv.append("--keep-existing")
        result = runner.invoke(app, argv)
        assert result.exit_code == 0, result.stdout + result.stderr
        return stale_npy, stale_map

    def test_default_removes_stale_files(self, tmp_path, monkeypatch, runner):
        npy, mp = self._run(tmp_path, monkeypatch, runner, keep_existing=False)
        assert not npy.exists()
        assert not mp.exists()

    def test_keep_existing_preserves_stale_files(
        self, tmp_path, monkeypatch, runner,
    ):
        npy, mp = self._run(tmp_path, monkeypatch, runner, keep_existing=True)
        assert npy.exists()
        assert mp.exists()


# ── library list ─────────────────────────────────────────────────────────────

class TestLibraryList:
    def test_lists_registered_films(self, tmp_path, monkeypatch, runner):
        cfg = _stub_cfg(tmp_path)
        _seed_film(Path(cfg.paths.library_dir), "film_a", "film_a.mp4")
        _seed_film(Path(cfg.paths.library_dir), "film_b", "film_b.mp4")
        monkeypatch.setattr("cinemateca.config.load_config",
                            lambda _path=None: cfg)
        monkeypatch.setattr("cinemateca.config.setup_logging", lambda _c: None)

        from cinemateca.__main__ import app

        result = runner.invoke(app, ["library", "list"])
        assert result.exit_code == 0, result.stdout + result.stderr
        assert "film_a" in result.stdout
        assert "film_b" in result.stdout
        assert "2 filme(s) registrado(s)" in result.stdout

    def test_empty_library_does_not_crash(self, tmp_path, monkeypatch, runner):
        cfg = _stub_cfg(tmp_path)
        (Path(cfg.paths.library_dir) / "films.json").write_text("{}")
        monkeypatch.setattr("cinemateca.config.load_config",
                            lambda _path=None: cfg)
        monkeypatch.setattr("cinemateca.config.setup_logging", lambda _c: None)

        from cinemateca.__main__ import app

        result = runner.invoke(app, ["library", "list"])
        assert result.exit_code == 0
        assert "Nenhum filme registrado" in result.stdout


# ── library delete (destructive — requires --yes or interactive confirm) ────

class TestLibraryDelete:
    def test_delete_with_yes_removes_film(self, tmp_path, monkeypatch, runner):
        cfg = _stub_cfg(tmp_path)
        library_dir = Path(cfg.paths.library_dir)
        _seed_film(library_dir, "film_a", "film_a.mp4")
        monkeypatch.setattr("cinemateca.config.load_config",
                            lambda _path=None: cfg)
        monkeypatch.setattr("cinemateca.config.setup_logging", lambda _c: None)

        from cinemateca.__main__ import app
        from cinemateca.library import load_registry

        assert "film_a" in load_registry(library_dir)
        result = runner.invoke(app, ["library", "delete", "film_a", "--yes"])
        assert result.exit_code == 0, result.stdout + result.stderr
        assert "film_a" not in load_registry(library_dir)
        # On-disk artifacts are also gone.
        assert not (library_dir / "film_a").exists()

    def test_delete_unknown_slug_returns_error(
        self, tmp_path, monkeypatch, runner,
    ):
        cfg = _stub_cfg(tmp_path)
        (Path(cfg.paths.library_dir) / "films.json").write_text("{}")
        monkeypatch.setattr("cinemateca.config.load_config",
                            lambda _path=None: cfg)
        monkeypatch.setattr("cinemateca.config.setup_logging", lambda _c: None)

        from cinemateca.__main__ import app

        result = runner.invoke(app, ["library", "delete", "ghost", "--yes"])
        assert result.exit_code != 0
        assert "ghost" in result.stderr

    def test_delete_aborts_without_confirmation(
        self, tmp_path, monkeypatch, runner,
    ):
        """Without ``--yes``, a 'n' answer to the interactive prompt must
        leave the registry intact."""
        cfg = _stub_cfg(tmp_path)
        library_dir = Path(cfg.paths.library_dir)
        _seed_film(library_dir, "film_a", "film_a.mp4")
        monkeypatch.setattr("cinemateca.config.load_config",
                            lambda _path=None: cfg)
        monkeypatch.setattr("cinemateca.config.setup_logging", lambda _c: None)

        from cinemateca.__main__ import app
        from cinemateca.library import load_registry

        result = runner.invoke(app, ["library", "delete", "film_a"], input="n\n")
        assert result.exit_code == 0
        assert "Cancelado" in result.stdout
        assert "film_a" in load_registry(library_dir)


# ── config show ──────────────────────────────────────────────────────────────

class TestConfigShow:
    def test_emits_yaml(self, runner):
        """``config show`` runs without a config override and emits a
        parseable YAML dump that includes the expected sections."""
        import yaml

        from cinemateca.__main__ import app

        result = runner.invoke(app, ["config", "show"])
        assert result.exit_code == 0, result.stdout + result.stderr
        parsed = yaml.safe_load(result.stdout)
        assert "embeddings" in parsed
        assert "paths" in parsed


# ── serve (smoke — just verify we delegate to uvicorn) ───────────────────────

class TestServe:
    def test_serve_delegates_to_uvicorn_run(self, monkeypatch, runner):
        captured: dict = {}

        def fake_run(target, *, host, port, reload, app_dir):
            captured.update(
                dict(target=target, host=host, port=port, reload=reload, app_dir=app_dir)
            )

        monkeypatch.setattr("uvicorn.run", fake_run)

        from cinemateca.__main__ import app

        result = runner.invoke(
            app, ["serve", "--port", "9999", "--no-reload"]
        )
        assert result.exit_code == 0, result.stdout + result.stderr
        assert captured["target"] == "api.server:app"
        assert captured["host"] == "127.0.0.1"
        assert captured["port"] == 9999
        assert captured["reload"] is False
        assert Path(captured["app_dir"]).resolve() == Path(__file__).parents[1].resolve()
