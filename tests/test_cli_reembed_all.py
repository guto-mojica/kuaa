"""Tests for ``cinemateca reembed-all`` CLI subcommand.

The key invariant: it must drive :class:`CatalogPipeline` with the
**registered** slug, never the filename-derived one. The bare
``process --video <path>`` form computes the slug from the filename,
which silently registered a stray ``mazzaropi-jeca_tatu_paixo_flix``
film during the Phase-1 density rebuild because the existing slug was
``jeca_tatu`` but the raw filename slugified to a different string.

These tests cover:
  * ``reembed-all`` walks ``films.json`` and instantiates the pipeline
    with each film's registered slug.
  * ``--only <slug>`` filters to a subset.
  * ``--only`` with an unknown slug exits 1.
  * Empty registry exits 1 with a clear message.
  * Default behavior removes the stale ``.npy`` before running;
    ``--keep-existing`` does not.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _seed_film(library_dir: Path, slug: str, raw_filename: str) -> Path:
    """Register a film and create the directories the CLI looks for."""
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
    raw_path.touch()  # placeholder; the test stubs out the real pipeline
    return raw_path


@pytest.fixture()
def reembed_env(tmp_path, monkeypatch):
    """Two registered films + a config that points at the temp library
    and stubs out the heavy bits (CatalogPipeline + setup_logging)."""
    library_dir = tmp_path / "library"
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    _seed_film(library_dir, "film_a", "film_a.mp4")
    _seed_film(library_dir, "film_b", "film_b.mp4")

    # Stub config: only the fields the CLI actually reads.
    fake_cfg = SimpleNamespace(
        paths=SimpleNamespace(library_dir=str(library_dir),
                              raw_dir=str(raw_dir)),
        pipeline=SimpleNamespace(steps=SimpleNamespace(
            frame_extraction=False, scene_detection=False,
            visual_analysis=False, embeddings=False, llm_description=False,
        )),
    )

    # Capture pipeline invocations.
    invocations: list[dict] = []

    class FakePipeline:
        def __init__(self, cfg, slug):
            self.cfg = cfg
            self.slug = slug

        def run(self, video_path):
            invocations.append({"slug": self.slug, "video": str(video_path)})
            return SimpleNamespace(success=True, total_duration_s=0.5)

    monkeypatch.setattr("cinemateca.__main__._print_banner", lambda: None)
    monkeypatch.setattr("cinemateca.config.load_config",
                        lambda _path=None: fake_cfg)
    monkeypatch.setattr("cinemateca.config.setup_logging", lambda _cfg: None)
    monkeypatch.setattr("cinemateca.pipeline.CatalogPipeline", FakePipeline)

    return SimpleNamespace(
        library_dir=library_dir,
        invocations=invocations,
        fake_cfg=fake_cfg,
    )


# ── Slug-from-registry invariant ─────────────────────────────────────────────

class TestReembedAllUsesRegisteredSlug:
    """The CLI must drive the pipeline with the registered slug, never
    a slug computed from the filename. This is the property whose
    absence (during Phase 1) silently registered an empty stray film.
    """

    def test_calls_pipeline_once_per_registered_film(self, reembed_env):
        from cinemateca.__main__ import cmd_reembed_all

        args = SimpleNamespace(
            config=None, steps="embeddings", only=None, keep_existing=False,
        )
        rc = cmd_reembed_all(args)
        assert rc == 0
        slugs_called = sorted(inv["slug"] for inv in reembed_env.invocations)
        assert slugs_called == ["film_a", "film_b"]

    def test_pipeline_slug_matches_registry_even_if_filename_drifts(
        self, tmp_path, monkeypatch
    ):
        """A registered slug ('canonical') with a filename whose
        slugified stem differs ('drifty-name') must invoke the pipeline
        with 'canonical', not 'drifty-name'."""
        library_dir = tmp_path / "library"
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        # raw filename slugifies to 'drifty-name', but registered slug is 'canonical'.
        _seed_film(library_dir, "canonical", "Drifty-Name.mp4")

        fake_cfg = SimpleNamespace(
            paths=SimpleNamespace(library_dir=str(library_dir),
                                  raw_dir=str(raw_dir)),
            pipeline=SimpleNamespace(steps=SimpleNamespace(
                frame_extraction=False, scene_detection=False,
                visual_analysis=False, embeddings=False, llm_description=False,
            )),
        )
        invocations: list[dict] = []

        class FakePipeline:
            def __init__(self, cfg, slug):
                self.slug = slug
            def run(self, video_path):
                invocations.append({"slug": self.slug})
                return SimpleNamespace(success=True, total_duration_s=0.0)

        monkeypatch.setattr("cinemateca.__main__._print_banner", lambda: None)
        monkeypatch.setattr("cinemateca.config.load_config",
                            lambda _path=None: fake_cfg)
        monkeypatch.setattr("cinemateca.config.setup_logging", lambda _cfg: None)
        monkeypatch.setattr("cinemateca.pipeline.CatalogPipeline", FakePipeline)

        from cinemateca.__main__ import cmd_reembed_all
        args = SimpleNamespace(
            config=None, steps="embeddings", only=None, keep_existing=False,
        )
        rc = cmd_reembed_all(args)
        assert rc == 0
        # If we used the filename, this would be "drifty-name" — the
        # exact bug Phase-1 hit. Lock it down: must be the registered slug.
        assert invocations == [{"slug": "canonical"}]


# ── Filter and edge cases ────────────────────────────────────────────────────

class TestReembedAllFilters:
    def test_only_filters_to_specified_slug(self, reembed_env):
        from cinemateca.__main__ import cmd_reembed_all

        args = SimpleNamespace(
            config=None, steps="embeddings",
            only=["film_a"], keep_existing=False,
        )
        rc = cmd_reembed_all(args)
        assert rc == 0
        assert [inv["slug"] for inv in reembed_env.invocations] == ["film_a"]

    def test_only_unknown_slug_returns_error(self, reembed_env, capsys):
        from cinemateca.__main__ import cmd_reembed_all

        args = SimpleNamespace(
            config=None, steps="embeddings",
            only=["does_not_exist"], keep_existing=False,
        )
        rc = cmd_reembed_all(args)
        assert rc == 1
        assert reembed_env.invocations == []
        err = capsys.readouterr().err
        assert "does_not_exist" in err

    def test_empty_registry_returns_error(self, tmp_path, monkeypatch, capsys):
        library_dir = tmp_path / "library"
        library_dir.mkdir()
        (library_dir / "films.json").write_text("{}")
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()

        fake_cfg = SimpleNamespace(
            paths=SimpleNamespace(library_dir=str(library_dir),
                                  raw_dir=str(raw_dir)),
            pipeline=SimpleNamespace(steps=SimpleNamespace(
                frame_extraction=False, scene_detection=False,
                visual_analysis=False, embeddings=False, llm_description=False,
            )),
        )
        monkeypatch.setattr("cinemateca.__main__._print_banner", lambda: None)
        monkeypatch.setattr("cinemateca.config.load_config",
                            lambda _path=None: fake_cfg)
        monkeypatch.setattr("cinemateca.config.setup_logging", lambda _cfg: None)

        from cinemateca.__main__ import cmd_reembed_all
        args = SimpleNamespace(
            config=None, steps="embeddings", only=None, keep_existing=False,
        )
        rc = cmd_reembed_all(args)
        assert rc == 1
        assert "Nenhum filme registrado" in capsys.readouterr().err


# ── Stale-file cleanup ───────────────────────────────────────────────────────

class TestReembedAllStaleCleanup:
    """Default behavior deletes existing .npy / index_mapping.json so
    the pipeline's ``skip_existing`` doesn't silently no-op the
    embeddings step. ``--keep-existing`` opts out."""

    def _setup(self, tmp_path, monkeypatch, keep_existing: bool):
        library_dir = tmp_path / "library"
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        _seed_film(library_dir, "film_a", "film_a.mp4")
        emb_dir = library_dir / "film_a" / "embeddings"
        stale_npy = emb_dir / "keyframe_embeddings.npy"
        stale_map = emb_dir / "index_mapping.json"
        stale_npy.write_bytes(b"stale")
        stale_map.write_text("{}")

        fake_cfg = SimpleNamespace(
            paths=SimpleNamespace(library_dir=str(library_dir),
                                  raw_dir=str(raw_dir)),
            pipeline=SimpleNamespace(steps=SimpleNamespace(
                frame_extraction=False, scene_detection=False,
                visual_analysis=False, embeddings=False, llm_description=False,
            )),
        )

        class FakePipeline:
            def __init__(self, cfg, slug): pass
            def run(self, _v):
                return SimpleNamespace(success=True, total_duration_s=0.0)

        monkeypatch.setattr("cinemateca.__main__._print_banner", lambda: None)
        monkeypatch.setattr("cinemateca.config.load_config",
                            lambda _path=None: fake_cfg)
        monkeypatch.setattr("cinemateca.config.setup_logging", lambda _cfg: None)
        monkeypatch.setattr("cinemateca.pipeline.CatalogPipeline", FakePipeline)

        from cinemateca.__main__ import cmd_reembed_all
        args = SimpleNamespace(
            config=None, steps="embeddings", only=None,
            keep_existing=keep_existing,
        )
        cmd_reembed_all(args)
        return stale_npy, stale_map

    def test_default_removes_stale_files(self, tmp_path, monkeypatch):
        stale_npy, stale_map = self._setup(tmp_path, monkeypatch,
                                            keep_existing=False)
        assert not stale_npy.exists(), "stale .npy must be cleared by default"
        assert not stale_map.exists(), "stale mapping must be cleared by default"

    def test_keep_existing_preserves_stale_files(self, tmp_path, monkeypatch):
        stale_npy, stale_map = self._setup(tmp_path, monkeypatch,
                                            keep_existing=True)
        assert stale_npy.exists(), "--keep-existing must NOT clear .npy"
        assert stale_map.exists(), "--keep-existing must NOT clear mapping"
