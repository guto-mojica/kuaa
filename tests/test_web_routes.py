"""
tests/test_web_routes.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Regression / characterization tests for the FastAPI web layer (v0.3.0).

This module is part of Phase 0 of the FastAPI regression-recovery effort.
Its purpose is NOT to assert the app is correct — it is to *document and
reproduce* the bugs that the recovery plan will fix in later phases.

Three groups of tests live here:

1. Empty-data smoke tests — these PASS today. They prove the routes
   respond at all when the data directory is empty (no GPU, no model,
   no video, no network). They are the safety net for the refactors.

2. ``xfail(strict=True)`` bug-reproduction tests — these FAIL today
   (reported as ``xfailed``). Each captures one verified defect:
     * ``TestFullPageContextDivergence`` — full-page routes (/scenes,
       /processing, /search, /annotate) render via ``base.html`` with a
       context that is missing keys the tab partials need, so they do
       NOT match the corresponding ``/tab/*`` output.
     * ``TestProcessingSplitFilterCrash`` — ``processing_job.html`` uses
       a non-existent Jinja ``split`` filter, so rendering any active
       job raises ``TemplateAssertionError``.
   When a later phase fixes the bug, ``strict=True`` flips the test to a
   hard failure (XPASS -> failed), forcing the fixer to delete the
   ``xfail`` marker and convert it to an ordinary passing assertion.

All tests use an isolated temp config so the repository ``data/``
directory is never read or written.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Allow `import cinemateca...` without an editable install (mirrors test_smoke).
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
# Allow `import api...` — the repo root holds the `api` package and is not
# otherwise on sys.path under pytest's default rootdir-based collection.
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Isolated app fixture ──────────────────────────────────────────────────────

@pytest.fixture()
def client(tmp_path, monkeypatch):
    """
    A TestClient whose config points every data path at empty temp
    directories. The routes call ``api.deps.get_config()`` directly
    (not via ``Depends``), and each route module imported the symbol
    into its own namespace, so we patch every binding and clear the
    ``lru_cache``.
    """
    from cinemateca.config import load_config

    # Real project root so default.yaml resolves; paths rebased to tmp.
    cfg = load_config(project_root=tmp_path)
    for name in (
        "data_dir",
        "raw_dir",
        "frames_dir",
        "metadata_dir",
        "embeddings_dir",
        "models_dir",
        "outputs_dir",
        "logs_dir",
    ):
        d = tmp_path / name
        d.mkdir(parents=True, exist_ok=True)
        setattr(cfg.paths, name, d)

    import api.deps as deps

    deps.get_config.cache_clear()
    monkeypatch.setattr(deps, "get_config", lambda: cfg)

    # Each route module did `from api.deps import get_config`, binding a
    # local name. Patch all of them so no route reaches the real data/.
    import api.server as server
    from api.routes import about, annotate, processing, scenes, search

    for mod in (server, scenes, search, annotate, processing):
        if hasattr(mod, "get_config"):
            monkeypatch.setattr(mod, "get_config", lambda: cfg)

    # about.py imports only make_ctx, no get_config — nothing to patch.
    _ = about

    # Reset the in-memory job registry so processing tests are hermetic.
    import api.jobs as jobs

    monkeypatch.setattr(jobs, "_jobs", {})

    from api.server import app

    with TestClient(app) as c:
        # The default locale is pt_BR (api/deps.make_ctx), whose catalog
        # translates UI strings to Portuguese. The `en` catalog has empty
        # msgstr entries, so gettext falls back to the English msgid —
        # i.e. the literal source strings the templates were written with
        # and that these tests assert on. Pin `en` for stable markers.
        c.cookies.set("locale", "en")
        yield c


@pytest.fixture()
def inject_job(monkeypatch):
    """Insert one running JobState into the registry and return it."""
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


# ── Group 1: empty-data smoke tests (must PASS) ───────────────────────────────

FULL_PAGES = ["/", "/search", "/scenes", "/annotate", "/processing"]
TAB_FRAGMENTS = ["/tab/search", "/tab/scenes", "/tab/annotate", "/tab/processing"]


@pytest.mark.parametrize("path", FULL_PAGES)
def test_full_page_routes_respond(client, path):
    """Full-page routes return 200 with the base shell rendered."""
    r = client.get(path)
    assert r.status_code == 200, r.text[:500]
    assert "<!DOCTYPE html>" in r.text
    assert 'id="tab-content"' in r.text


@pytest.mark.parametrize("path", TAB_FRAGMENTS)
def test_tab_fragment_routes_respond(client, path):
    """`/tab/*` partials return 200 and a tab panel (no full HTML doc)."""
    r = client.get(path)
    assert r.status_code == 200, r.text[:500]
    assert 'class="tab-panel"' in r.text
    assert "<!DOCTYPE html>" not in r.text


def test_about_modal_responds(client):
    """/api/about renders the about modal partial."""
    r = client.get("/api/about")
    assert r.status_code == 200, r.text[:500]
    assert "0.3.0" in r.text


def test_tab_processing_empty_has_no_active_jobs(client):
    """With no jobs the processing tab shows the empty-state line."""
    r = client.get("/tab/processing")
    assert r.status_code == 200, r.text[:500]
    assert "No active jobs." in r.text


# ── Group 2a: full-page vs tab context divergence (xfail, strict) ─────────────

class TestFullPageContextDivergence:
    """
    ``_base_page()`` in api/server.py supplies only
    ``active_tab, processing_jobs, films, selected_slug``. ``base.html``
    then ``{% include %}``s the tab partials, which need far more
    context. So a direct GET of a full-page route renders a degraded /
    incomplete panel compared to the dedicated ``/tab/*`` route.

    Each test below asserts the full-page response contains a specific
    marker that the matching ``/tab/*`` response contains. They FAIL
    today (the marker is absent from the full page) — that absence *is*
    the documented bug.
    """

    @pytest.mark.xfail(
        reason="Phase 0 documents bug; fixed in Phase 1a "
        "(/scenes full page lacks no_data -> empty-state markup)",
        strict=True,
    )
    def test_scenes_full_page_matches_tab_empty_state(self, client):
        tab = client.get("/tab/scenes")
        assert tab.status_code == 200
        # /tab/scenes with empty data renders the scene-detection hint.
        marker = "Run the pipeline with the Scene Detection step first."
        assert marker in tab.text, "precondition: tab must show empty state"

        full = client.get("/scenes")
        assert full.status_code == 200
        # BUG: /scenes is missing `no_data`, so the {% if no_data %}
        # branch is falsy and the empty-state hint never renders.
        assert marker in full.text

    @pytest.mark.xfail(
        reason="Phase 0 documents bug; fixed in Phase 1a "
        "(/processing full page lacks step_defs -> steps checklist empty)",
        strict=True,
    )
    def test_processing_full_page_steps_checklist_not_empty(self, client):
        # NOTE: the "No active jobs." line is NOT a usable discriminator
        # here — on the full page `jobs` is *undefined*, which Jinja
        # treats as falsy, so the {% else %} branch renders the same
        # text as the tab. The honest, discriminating symptom of the
        # missing context is the empty steps checklist below.
        full = client.get("/processing")
        assert full.status_code == 200
        # BUG: /processing is missing `step_defs`, so the
        # {% for name, label in step_defs %} loop iterates an undefined
        # (Jinja silently yields nothing) and the checklist div is empty.
        assert '<div class="pipeline-steps-check">' in full.text
        body = full.text.split('pipeline-steps-check">', 1)[1].split("</div>", 1)[0]
        assert body.strip() != "", "steps checklist rendered empty"

    @pytest.mark.xfail(
        reason="Phase 0 documents bug; fixed in Phase 1a "
        "(/processing full page lacks step_defs -> steps checklist gone)",
        strict=True,
    )
    def test_processing_full_page_renders_step_checklist(self, client):
        tab = client.get("/tab/processing")
        full = client.get("/processing")
        assert tab.status_code == 200 and full.status_code == 200
        # The Steps label only appears when step_defs drives the loop;
        # /tab/processing supplies step_defs, /processing does not.
        assert tab.text.count("tag-pill") > 0, "precondition"
        assert full.text.count("tag-pill") == tab.text.count("tag-pill")

    @pytest.mark.xfail(
        reason="Phase 0 documents bug; fixed in Phase 1a "
        "(/annotate full page lacks no_data/all_done/total/etc.)",
        strict=True,
    )
    def test_annotate_full_page_matches_tab(self, client):
        tab = client.get("/tab/annotate")
        assert tab.status_code == 200
        # Empty data -> annotate partial shows the no_data hint.
        marker = "Run the Scene Detection step first."
        assert marker in tab.text, "precondition: tab must show empty state"

        full = client.get("/annotate")
        assert full.status_code == 200
        # BUG: /annotate is missing `no_data` (and all_done/total/...),
        # so the partial renders an undefined-variable / wrong branch.
        assert marker in full.text


# ── Group 2b: Processing `split` filter crash (xfail, strict) ─────────────────

class TestProcessingSplitFilterCrash:
    """
    ``web/templates/partials/processing_job.html`` uses
    ``{{ job.video_path | replace('\\\\','/') | split('/') | last }}``.
    Jinja2 has no built-in ``split`` filter and the app's Jinja env
    (api/templates.py) registers none. Rendering ANY active job raises
    ``jinja2.exceptions.TemplateAssertionError: No filter named 'split'``.

    ``/tab/processing`` includes ``processing_job.html`` for every active
    job, so injecting one job and GETting the tab must trip the crash.
    """

    @pytest.mark.xfail(
        reason="Phase 0 documents bug; fixed in Phase 1b "
        "(processing_job.html uses non-existent `split` filter)",
        strict=True,
    )
    def test_tab_processing_with_active_job_renders(self, client, inject_job):
        inject_job()
        r = client.get("/tab/processing")
        # BUG: this raises TemplateAssertionError (No filter named
        # 'split') inside the TestClient, surfacing as a 500 / exception.
        assert r.status_code == 200, r.text[:500]
        # If the split filter existed, the basename would appear.
        assert "jeca_tatu.mp4" in r.text

    def test_split_filter_is_genuinely_absent(self):
        """
        Sanity probe (NOT xfail): proves the crash is a real missing
        filter, so the xfail above is not passing/failing for an
        unrelated reason. Rendering the template directly must raise
        with the exact 'No filter named' message.
        """
        import jinja2

        from api.jobs import STEP_DEFS, JobState, StepInfo
        from api.templates import templates

        job = JobState(
            id="probe",
            video_path="data/raw/jeca_tatu.mp4",
            steps=[StepInfo(name=n, label=lbl) for n, lbl in STEP_DEFS],
        )
        # Jinja resolves filter names at *compile* time, so the
        # TemplateAssertionError fires on get_template(), before render.
        with pytest.raises(jinja2.exceptions.TemplateAssertionError) as exc:
            tmpl = templates.env.get_template("partials/processing_job.html")
            tmpl.render(job=job)
        assert "No filter named 'split'" in str(exc.value)
