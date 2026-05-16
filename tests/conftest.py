"""
tests/conftest.py
~~~~~~~~~~~~~~~~~~
Shared test isolation machinery for the FastAPI web layer.

This file is the single source of truth for the temp-config / hermetic
``TestClient`` pattern. Before Phase 2 the same ~40 lines of config
rebasing + per-route-module ``get_config`` monkeypatching + job-registry
reset were copy-pasted into ``test_web_routes.py``, ``test_scene_id_filtering.py``
and ``test_sse.py``. The duplication had already drifted (the structural
guard comment in test_web_routes documents a prior drift). Phase 2
extracts it here so every web test — old and new — shares one isolation
path that provably never touches the repository ``data/`` directory.

Fixtures provided:

  * ``tmp_config``  — a ``Config`` whose every ``paths.*`` entry is a
    fresh tmp subdir; also clears the relevant ``lru_cache``s and
    rebinds ``get_config`` in every route module that imported it.
    The route modules are *discovered dynamically* (``pkgutil`` over
    the ``api.routes`` package, plus ``api.server``) rather than read
    off a hand-maintained list, so a newly added ``api/routes/*.py``
    that imports ``get_config`` is patched + asserted automatically;
    no fixture edit is needed and none can be forgotten.
  * ``client``      — an empty-data ``TestClient`` (locale pinned ``en``)
    built on ``tmp_config``.
  * ``seed_metadata`` — FACTORY fixture (mirrors ``inject_job``):
    returns a callable that writes a dataset into the temp
    metadata/frames dirs and returns the paths. Called with no
    arguments it produces the EXACT historical default dataset
    (2 scenes, fake keyframe file, LLM tag index with INT ids,
    manual annotations with STR keys, one visual-analysis record)
    byte-identically, so every pre-existing test that used the old
    fixed-shape fixture keeps passing unchanged. Callers that need a
    different shape pass explicit ``scenes`` / ``llm_tags`` /
    ``manual`` / ``descriptions`` / ``visual`` specs (Phase 3a-c/5).
  * ``inject_job``  — insert one running ``JobState`` into the registry.

``tmp_config`` additionally asserts, while the temp config is active,
that every ``cfg.paths.*`` entry resolves OUTSIDE the repo root —
converting a non-hermetic regression (a path that escaped the sandbox)
into an immediate, in-test failure. (This is done in-fixture rather than
as an autouse teardown hook because pytest undoes ``monkeypatch`` before
autouse post-yield code runs, so a teardown check would always observe
the real cached config and false-positive.)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Mirror the sys.path bootstrap the legacy test modules each did inline,
# so `import cinemateca...` / `import api...` work without an editable
# install under pytest's rootdir-based collection.
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

REPO = _REPO_ROOT

_PATH_NAMES = (
    "data_dir",
    "raw_dir",
    "frames_dir",
    "metadata_dir",
    "embeddings_dir",
    "models_dir",
    "outputs_dir",
    "logs_dir",
)


# ── Core isolated-config fixture ──────────────────────────────────────────────

@pytest.fixture()
def tmp_config(tmp_path, monkeypatch):
    """A ``Config`` with every data path rebased into ``tmp_path``.

    The routes call ``api.deps.get_config()`` directly (not via
    ``Depends``), and each route module did ``from api.deps import
    get_config`` — binding a *local* name. So we patch every binding,
    clear the ``lru_cache``s, and reset the in-memory job registry.

    Returns the rebased ``Config`` object; tests rarely need it
    directly but ``seed_metadata`` and ``client`` build on it.
    """
    from cinemateca.config import load_config

    # Real project root so default.yaml resolves; paths rebased to tmp.
    cfg = load_config(project_root=tmp_path)
    for name in _PATH_NAMES:
        d = tmp_path / name
        d.mkdir(parents=True, exist_ok=True)
        setattr(cfg.paths, name, d)

    import api.deps as deps

    deps.get_config.cache_clear()
    monkeypatch.setattr(deps, "get_config", lambda: cfg)

    # search._load_index is @lru_cache(maxsize=1) keyed on the embeddings
    # dir path. It is NOT keyed on cfg identity, so a populated/corrupt
    # index loaded by one test would leak into the next. Clear it per
    # test. (The legacy fixtures never did this because no legacy test
    # exercised a populated index through the route — Phase 2 does.)
    import api.routes.search as search_route

    search_route._load_index.cache_clear()

    # Dynamic hermeticity guard. The routes do `from api.deps import
    # get_config`, binding a *local* name in each module; patching only
    # api.deps is not enough. Rather than trust a hand-maintained list
    # (which drifted before — see test_web_routes' guard comment), we
    # DISCOVER every module under the api.routes package plus api.server,
    # import each, and for every one that bound a `get_config` name we
    # rebind it AND assert it now resolves to the temp cfg. A new
    # api/routes/*.py that imports get_config is therefore covered
    # automatically; if discovery can't reach a module that needs it,
    # the assert below fires loudly at fixture-construction time instead
    # of letting that module silently read the real repo data/.
    import importlib
    import pkgutil

    import api.routes as routes_pkg
    import api.server as server

    mods = [server]
    for info in pkgutil.iter_modules(routes_pkg.__path__):
        mods.append(importlib.import_module(f"{routes_pkg.__name__}.{info.name}"))

    for mod in mods:
        if hasattr(mod, "get_config"):
            monkeypatch.setattr(mod, "get_config", lambda: cfg)
            assert mod.get_config() is cfg, (
                f"{mod.__name__}.get_config was not rebound to the temp "
                f"config — this module would read the real repo data/"
            )

    # Reset the in-memory job registry so processing tests are hermetic.
    import api.jobs as jobs

    monkeypatch.setattr(jobs, "_jobs", {})

    # Hermeticity guard, checked WHILE the temp config is active (not at
    # teardown — monkeypatch is undone before autouse post-yield runs, so
    # a teardown check would always see the real cached config and false-
    # positive). Every data path the routes will touch must resolve
    # OUTSIDE the repo root. This, plus the dynamic per-module
    # rebinding assertion above (every discovered api.routes module +
    # api.server that bound get_config is rebound and verified), is the
    # proof no test mutates the real data/.
    repo = REPO.resolve()
    for name in _PATH_NAMES:
        p = Path(getattr(cfg.paths, name)).resolve()
        assert repo not in p.parents and p != repo, (
            f"cfg.paths.{name} resolved to {p}, inside the repo root "
            f"{repo} — the temp-config sandbox did not rebase this path; "
            f"a test could mutate the real data/ directory"
        )

    return cfg


# ── TestClient fixtures ───────────────────────────────────────────────────────

def _make_client(cfg) -> TestClient:
    from api.server import app

    c = TestClient(app)
    c.__enter__()
    # Default locale is pt_BR (api/deps.make_ctx); the `en` catalog has
    # empty msgstr entries so gettext falls back to the English msgid —
    # the literal source strings the templates were written with and
    # that these tests assert on. Pin `en` for stable markers.
    c.cookies.set("locale", "en")
    return c


@pytest.fixture()
def client(tmp_config):
    """Empty-data ``TestClient`` on the isolated temp config."""
    c = _make_client(tmp_config)
    try:
        yield c
    finally:
        c.__exit__(None, None, None)


# ── Fixture-metadata builder ──────────────────────────────────────────────────

@pytest.fixture()
def seed_metadata(tmp_config):
    """Factory: write a dataset into the temp dirs, return its paths.

    Mirrors the ``inject_job`` factory pattern: the fixture yields a
    callable. Invoked with no arguments it writes the historical
    default dataset *byte-identically* (so every pre-existing test
    that did ``seed_metadata`` then requested ``client`` keeps passing
    unchanged — see ``_DEFAULT_*`` below, which are the literal values
    the old fixed fixture wrote). Callers needing a different dataset
    shape (Phase 3a-c/5) override any of ``scenes`` / ``llm_tags`` /
    ``manual`` / ``descriptions`` / ``visual``.

    Returns a dict of useful paths/objects. Schema choices match what
    the routes actually READ (verified against the route source), not
    necessarily the full producer schema:

      * ``keyframes_metadata.json`` — list of scene dicts. scenes.py /
        annotate.py read ``scene_id``, ``filepath``, ``timecode_start``,
        ``start_time_s``/``end_time_s``. INT scene_id (as the keyframe
        extractor emits).
      * one fake keyframe file is ``touch``ed on disk under the temp
        frames dir. No route opens it for these tests (img_url only
        resolves a /media URL via path math; no PIL load on the
        text/scene paths), so an empty placeholder is sufficient and
        keeps the test CLIP/PIL-free.
      * ``scene_tags.json`` — LLM inverted index, values are INT scene
        ids (LLMDescriber.build_tag_index appends ``record["scene_id"]``
        which is ``int(...)``; round-tripped through JSON they stay int
        list-values).
      * ``manual_annotations.json`` — JSON object => STRING scene-id
        keys. The filename is ``annotator.FILENAME`` and ``annotator``
        is what *reads* it (``annotator.load``); the STR-key shape and
        any lower-kebab tag normalization are produced by the SAVE
        ROUTE (``api/routes/annotate.py`` does
        ``ann[str(scene_id)] = [t.strip().lower()...]``), not by
        ``annotator.save`` which only ``json.dump``s the dict it gets.
        This seed writes that on-disk shape directly.
      * ``scene_descriptions.json`` — list with ``scene_id`` +
        ``description`` (read by scenes keyword filter / annotate llm).
      * ``visual_analysis.json`` — scenes.py reads ``scene_id``,
        ``environment.location``, ``environment.time_of_day``,
        ``num_faces`` (the per-scene flattened shape the route expects).
    """
    cfg = tmp_config
    meta_dir = Path(cfg.paths.metadata_dir)
    frames_dir = Path(cfg.paths.frames_dir)

    _SENTINEL = object()

    def _seed(
        *,
        scenes=_SENTINEL,
        llm_tags=_SENTINEL,
        manual=_SENTINEL,
        descriptions=_SENTINEL,
        visual=_SENTINEL,
        keyframe_file_name: str = "s351.jpg",
    ) -> dict:
        """Write the requested (or default) dataset; return path map.

        Each ``_SENTINEL`` argument falls back to the historical
        default value (verbatim from the pre-factory fixture). One
        keyframe file is always ``touch``ed so the default scene 351's
        ``filepath`` points at a real on-disk placeholder, exactly as
        before. Pass an arg explicitly (including ``None`` to skip
        writing that file) to override.
        """
        kf_file = frames_dir / keyframe_file_name
        kf_file.touch()  # placeholder; no route opens it on these paths

        _default_scenes = [
            {
                "scene_id": 351,
                "filepath": str(kf_file),
                "timecode_start": "00:01:23",
                "start_time_s": 83.0,
                "end_time_s": 90.0,
            },
            {
                "scene_id": 352,
                "filepath": "frames/s352.jpg",
                "timecode_start": "00:02:00",
                "start_time_s": 120.0,
                "end_time_s": 128.0,
            },
        ]
        # LLM tag index — INT scene ids (post build_tag_index + JSON round-trip).
        _default_llm = {"exterior": [351, 352], "dia": [351]}
        # Manual annotations — JSON object => STRING keys.
        _default_manual = {"352": ["manual-only", "noite"]}
        _default_desc = [
            {"scene_id": 351, "description": "a man walking outdoors at dawn"},
            {"scene_id": 352, "description": "interior office scene"},
        ]
        _default_visual = [
            {
                "scene_id": 351,
                "environment": {"location": "exterior", "time_of_day": "dia"},
                "num_faces": 2,
            }
        ]

        scenes_v = _default_scenes if scenes is _SENTINEL else scenes
        llm_v = _default_llm if llm_tags is _SENTINEL else llm_tags
        manual_v = _default_manual if manual is _SENTINEL else manual
        desc_v = _default_desc if descriptions is _SENTINEL else descriptions
        visual_v = _default_visual if visual is _SENTINEL else visual

        # ``None`` => deliberately omit that artefact file (e.g. to test
        # a partially-processed library); any other value is written.
        if scenes_v is not None:
            (meta_dir / "keyframes_metadata.json").write_text(json.dumps(scenes_v))
        if llm_v is not None:
            (meta_dir / "scene_tags.json").write_text(json.dumps(llm_v))
        if manual_v is not None:
            (meta_dir / "manual_annotations.json").write_text(json.dumps(manual_v))
        if desc_v is not None:
            (meta_dir / "scene_descriptions.json").write_text(json.dumps(desc_v))
        if visual_v is not None:
            (meta_dir / "visual_analysis.json").write_text(json.dumps(visual_v))

        scene_ids = (
            [s["scene_id"] for s in scenes_v if "scene_id" in s]
            if isinstance(scenes_v, list)
            else []
        )
        return {
            "cfg": cfg,
            "meta_dir": meta_dir,
            "frames_dir": frames_dir,
            "keyframe_file": kf_file,
            "manual_path": meta_dir / "manual_annotations.json",
            "scene_ids": scene_ids,
        }

    return _seed


# ── Job registry helper ───────────────────────────────────────────────────────

@pytest.fixture()
def inject_job():
    """Insert one running ``JobState`` into the registry and return it.

    Depends on nothing but the registry; tests that also need an
    isolated config simply request ``client`` too (which resets
    ``jobs._jobs`` via ``tmp_config``). ``test_sse.py`` builds its own
    lighter variant on this same registry contract.
    """
    import api.jobs as jobs

    def _inject(video_path: str = "data/raw/jeca_tatu.mp4"):
        job = jobs.JobState(
            id="testjob1",
            video_path=video_path,
            steps=[
                jobs.StepInfo(name=name, label=label)
                for name, label in jobs.STEP_DEFS
            ],
        )
        job.steps[0].state = "active"
        jobs._jobs[job.id] = job
        return job

    return _inject
