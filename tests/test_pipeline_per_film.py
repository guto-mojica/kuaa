"""
tests/test_pipeline_per_film.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
T11: Pipeline writes per-film paths.

Verifies that ``CatalogPipeline`` routes all outputs to
``data/library/<slug>/...`` when a slug is supplied, that it calls
``register_film`` on the first run and is idempotent on re-runs.

No real models run — ``_step_*`` are stubbed to return successful
``StepResult`` objects without touching video/CLIP/Torch.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from cinemateca.pipeline import CatalogPipeline, StepResult

# ── Helpers ───────────────────────────────────────────────────────────────────


def _fake_cfg(tmp_path: Path):
    """Minimal ``Config``-like namespace whose paths are rebased to tmp_path."""
    library_dir = tmp_path / "library"
    library_dir.mkdir(parents=True, exist_ok=True)

    paths = types.SimpleNamespace(
        library_dir=library_dir,
        # Legacy flat paths (must NOT be written to when slug is set).
        frames_dir=tmp_path / "legacy_frames",
        metadata_dir=tmp_path / "legacy_meta",
        embeddings_dir=tmp_path / "legacy_embeddings",
        raw_dir=tmp_path / "legacy_raw",
    )
    pipeline_ns = types.SimpleNamespace(
        skip_existing=False,
        stop_on_error=True,
        steps=types.SimpleNamespace(
            frame_extraction=True,
            scene_detection=True,
            visual_analysis=True,
            embeddings=True,
            llm_description=True,
            audio_extract=True,
            audio_embed=True,
        ),
    )
    embeddings_ns = types.SimpleNamespace(
        filename="keyframe_embeddings.npy",
        mapping_filename="keyframe_mapping.csv",
    )
    llm_ns = types.SimpleNamespace(
        descriptions_filename="scene_descriptions.json",
        tags_filename="scene_tags.json",
    )
    hardware_ns = types.SimpleNamespace(device="cpu")
    logging_ns = types.SimpleNamespace(level="WARNING", to_file=False, filename="app.log")
    paths.logs_dir = tmp_path / "logs"
    return types.SimpleNamespace(
        paths=paths,
        pipeline=pipeline_ns,
        embeddings=embeddings_ns,
        llm=llm_ns,
        hardware=hardware_ns,
        logging=logging_ns,
    )


def _stub_steps(pipeline: CatalogPipeline) -> None:
    """Replace all ``_step_*`` with no-op stubs that return success."""

    def _ok(name):
        def _stub(*_a, **_kw):
            return StepResult(name=name, success=True, duration_s=0.0)
        return _stub

    pipeline._step_frame_extraction = _ok("frame_extraction")
    pipeline._step_scene_detection = _ok("scene_detection")
    pipeline._step_visual_analysis = _ok("visual_analysis")
    pipeline._step_embeddings = _ok("embeddings")
    pipeline._step_llm_description = _ok("llm_description")
    pipeline._step_audio_extract = _ok("audio_extract")
    pipeline._step_audio_embed = _ok("audio_embed")


# ── slugify helper ────────────────────────────────────────────────────────────


def test_slugify_basic():
    """slugify converts stems to clean slugs."""
    from cinemateca.pipeline import slugify

    assert slugify("jeca_tatu") == "jeca_tatu"
    assert slugify("Jeca Tatu") == "jeca_tatu"
    assert slugify("My Film (1959)") == "my_film_1959"
    assert slugify("some-film") == "some-film"
    assert slugify("UPPERCASE") == "uppercase"


def test_slugify_strips_non_alnum():
    from cinemateca.pipeline import slugify

    assert slugify("hello world!") == "hello_world"
    assert slugify("../secret") == "secret"


def test_slugify_degenerate_inputs_return_empty():
    """Inputs that contain no slug-safe characters slugify to ''.

    Combined with the empty-slug guard in CatalogPipeline.__init__, this
    means such inputs raise loudly instead of producing a corrupt
    library_dir/'' film entry.
    """
    from cinemateca.pipeline import slugify

    assert slugify("") == ""
    assert slugify("!!!") == ""


def test_pipeline_empty_slug_raises(tmp_path):
    """Empty slug after sanitization → ValueError (not silent corruption)."""
    cfg = _fake_cfg(tmp_path)
    with pytest.raises(ValueError, match="Slug is empty"):
        CatalogPipeline(cfg, slug="")


def test_pipeline_does_not_register_on_partial_failure(tmp_path):
    """When a step fails (stop_on_error=False, result.success=False), the
    film is NOT registered — partial runs leave the registry untouched."""
    from cinemateca.library import load_registry

    cfg = _fake_cfg(tmp_path)
    p = CatalogPipeline(cfg, slug="partial")

    # Stub: frame_extraction succeeds, scene_detection fails, others succeed.
    # The pipeline's _step_* methods take (*_a, **_kw) per the _stub_steps
    # pattern in this file. result.success becomes False because of the
    # scene_detection failure.
    def _ok(name):
        def _stub(*_a, **_kw):
            return StepResult(name=name, success=True, duration_s=0.0)
        return _stub

    def _fail(name):
        def _stub(*_a, **_kw):
            return StepResult(name=name, success=False, duration_s=0.0)
        return _stub

    p._step_frame_extraction = _ok("frame_extraction")
    p._step_scene_detection = _fail("scene_detection")
    p._step_visual_analysis = _ok("visual_analysis")
    p._step_embeddings = _ok("embeddings")
    p._step_llm_description = _ok("llm_description")

    p.run(tmp_path / "video.mp4")

    registry = load_registry(cfg.paths.library_dir)
    assert "partial" not in registry, (
        "Pipeline must NOT register a film whose run failed"
    )


# ── Per-film path routing ─────────────────────────────────────────────────────


def test_pipeline_per_film_creates_subdirs(tmp_path):
    """Pipeline with a slug creates ``library/<slug>/`` subdirectories."""
    cfg = _fake_cfg(tmp_path)
    p = CatalogPipeline(cfg, slug="my_film")
    _stub_steps(p)

    p.run("video.mp4")

    slug_dir = cfg.paths.library_dir / "my_film"
    assert slug_dir.is_dir(), "Film slug directory should be created"
    assert (slug_dir / "metadata").is_dir()
    assert (slug_dir / "frames").is_dir()
    assert (slug_dir / "embeddings").is_dir()


def test_pipeline_per_film_does_not_write_to_flat_paths(tmp_path):
    """Per-film run must NOT touch the legacy flat data paths."""
    cfg = _fake_cfg(tmp_path)
    p = CatalogPipeline(cfg, slug="my_film")
    _stub_steps(p)

    p.run("video.mp4")

    # Legacy flat dirs must NOT have been created by this run.
    assert not (tmp_path / "legacy_frames").exists(), (
        "Pipeline should not write to flat legacy frames_dir when slug is set"
    )
    assert not (tmp_path / "legacy_meta").exists(), (
        "Pipeline should not write to flat legacy metadata_dir when slug is set"
    )
    assert not (tmp_path / "legacy_embeddings").exists(), (
        "Pipeline should not write to flat legacy embeddings_dir when slug is set"
    )


def test_pipeline_registers_film_in_films_json(tmp_path):
    """After a successful run the slug appears in ``films.json``."""
    from cinemateca.library import load_registry

    cfg = _fake_cfg(tmp_path)
    p = CatalogPipeline(cfg, slug="my_film")
    _stub_steps(p)

    p.run("data/library/my_film/raw/video.mp4")

    registry = load_registry(cfg.paths.library_dir)
    assert "my_film" in registry, "Slug must be registered in films.json after run"
    assert registry["my_film"]["raw_filename"] == "video.mp4"


def test_pipeline_register_is_idempotent(tmp_path):
    """Re-running on an already-registered slug must not raise."""
    cfg = _fake_cfg(tmp_path)

    p1 = CatalogPipeline(cfg, slug="my_film")
    _stub_steps(p1)
    result1 = p1.run("video.mp4")
    assert result1.success

    # Second run — same slug already registered.
    p2 = CatalogPipeline(cfg, slug="my_film")
    _stub_steps(p2)
    result2 = p2.run("video.mp4")
    assert result2.success, "Re-run on registered slug should succeed"


def test_pipeline_year_extraction_from_filename(tmp_path):
    """Year is extracted from the video stem when a 4-digit year is present."""
    from cinemateca.library import load_registry

    cfg = _fake_cfg(tmp_path)
    p = CatalogPipeline(cfg, slug="jeca_tatu_1959")
    _stub_steps(p)

    p.run("data/raw/jeca_tatu_1959.mp4")

    registry = load_registry(cfg.paths.library_dir)
    assert registry["jeca_tatu_1959"]["year"] == 1959


def test_pipeline_no_slug_uses_legacy_paths(tmp_path):
    """Without a slug the pipeline still uses cfg.paths.* (legacy flat layout)."""
    cfg = _fake_cfg(tmp_path)
    # Create the legacy dirs so path-accessing code doesn't fail.
    cfg.paths.frames_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.metadata_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.embeddings_dir.mkdir(parents=True, exist_ok=True)

    p = CatalogPipeline(cfg)  # no slug
    _stub_steps(p)

    result = p.run("video.mp4")
    assert result.success


# ── CLI integration ───────────────────────────────────────────────────────────


def test_cli_slug_default_is_slugified_stem(tmp_path, monkeypatch):
    """``--slug`` defaults to slugified ``Path(--video).stem``."""
    import sys

    from cinemateca import __main__ as main_mod

    cfg = _fake_cfg(tmp_path)

    # Patch load_config where __main__ imports it from at call time.
    import cinemateca.config as cfg_mod
    monkeypatch.setattr(cfg_mod, "load_config", lambda *a, **kw: cfg)

    # Capture the slug that CatalogPipeline is constructed with.
    captured: dict = {}
    original_init = CatalogPipeline.__init__
    original_run = CatalogPipeline.run

    def fake_init(self, c, slug=None):
        captured["slug"] = slug
        original_init(self, c, slug=slug)

    def fake_run(self, video_path):
        _stub_steps(self)
        return original_run(self, video_path)

    monkeypatch.setattr(CatalogPipeline, "__init__", fake_init)
    monkeypatch.setattr(CatalogPipeline, "run", fake_run)

    test_video = tmp_path / "my_film_2001.mp4"
    test_video.touch()

    # Typer CLI: ``video`` is a positional argument; old argparse form
    # ``--video <path>`` was retired in the unified-CLI refactor.
    monkeypatch.setattr(sys, "argv", [
        "cinemateca", "process", str(test_video),
    ])
    with pytest.raises(SystemExit):
        main_mod.main()

    assert captured.get("slug") == "my_film_2001", (
        f"Expected slug 'my_film_2001', got {captured.get('slug')!r}"
    )


def test_cli_explicit_slug_is_forwarded(tmp_path, monkeypatch):
    """An explicit ``--slug`` is passed through to ``CatalogPipeline``."""
    import sys

    from cinemateca import __main__ as main_mod

    cfg = _fake_cfg(tmp_path)

    import cinemateca.config as cfg_mod
    monkeypatch.setattr(cfg_mod, "load_config", lambda *a, **kw: cfg)

    captured: dict = {}
    original_init = CatalogPipeline.__init__
    original_run = CatalogPipeline.run

    def fake_init(self, c, slug=None):
        captured["slug"] = slug
        original_init(self, c, slug=slug)

    def fake_run(self, video_path):
        _stub_steps(self)
        return original_run(self, video_path)

    monkeypatch.setattr(CatalogPipeline, "__init__", fake_init)
    monkeypatch.setattr(CatalogPipeline, "run", fake_run)

    test_video = tmp_path / "film.mp4"
    test_video.touch()

    monkeypatch.setattr(sys, "argv", [
        "cinemateca", "process", str(test_video), "--slug", "custom_slug",
    ])
    with pytest.raises(SystemExit):
        main_mod.main()

    assert captured.get("slug") == "custom_slug"
