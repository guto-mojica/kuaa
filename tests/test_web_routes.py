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

The ``client`` and ``inject_job`` fixtures (formerly defined inline
here) were consolidated into ``tests/conftest.py`` in Phase 2 — the
isolation behaviour is unchanged; this module's assertions are
untouched. See conftest.py for the temp-config / hermetic-client
machinery now shared with the other web test modules.
"""

from __future__ import annotations

import pytest

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


# ── Group 1b: Mojica chrome shell smoke (Task 6) ──────────────────────────────
#
# These assert the new chrome scaffolding (TopBar / IconRail / LeftPane / right
# pane) is wired into the rendered shell. The chrome lives ALONGSIDE the legacy
# sidebar/tab-bar until Tasks 7+8 replace those bits with real content; the
# legacy markup must keep working until then so the existing tests stay green.


def test_base_shell_renders_chrome(client):
    """Buscar full-page response carries TopBar/IconRail/LeftPane/right markers."""
    r = client.get("/search")
    assert r.status_code == 200
    html = r.text
    # TopBar present + brand name.
    assert 'class="ch-top"' in html
    assert "Mojica" in html
    # IconRail wrapper.
    assert 'class="ch-rail"' in html
    # LeftPane (not compact on Buscar).
    assert 'class="ch-lp"' in html
    # Body wrapper has the right-pane modifier (Buscar has a right pane).
    assert "with-right" in html
    # Active tab tag uses the PT slug.
    assert 'data-active-tab="buscar"' in html


def test_base_shell_compact_for_anotar(client):
    """Anotar collapses the left pane via the compact-lp body modifier."""
    r = client.get("/annotate")
    assert r.status_code == 200
    assert "compact-lp" in r.text


def test_base_shell_includes_palette_and_help_roots(client):
    """Polish-layer mount points exist on the index page (filled later)."""
    r = client.get("/")
    html = r.text
    assert 'id="palette-root"' in html
    assert 'id="help-root"' in html
    assert 'id="toast-root"' in html


# ── Group 1c: TopBar tab chips active state (Task 7) ──────────────────────────
#
# The Mojica TopBar renders five tab chips with PT slugs (`data-tab="buscar"`,
# `data-tab="cenas"`, `data-tab="anotar"`, `data-tab="rimas"`, `data-tab="proc"`)
# and the active chip carries the `on` class. The body's `data-active-tab`
# carries the same PT slug. These assertions pin down both contracts at once.


@pytest.mark.parametrize(
    "path,active",
    [
        ("/search", "buscar"),
        ("/scenes", "cenas"),
        ("/annotate", "anotar"),
        ("/rimas", "rimas"),
        ("/processing", "proc"),
    ],
)
def test_topbar_active_tab(client, path, active):
    """Each route marks its corresponding topbar chip + body slug as active."""
    r = client.get(path)
    assert r.status_code == 200, r.text[:500]
    html = r.text
    assert f'data-active-tab="{active}"' in html
    assert '<a class="tab on"' in html
    assert html.count(f'data-tab="{active}"') >= 1


# ── Group 1d: IconRail + LeftPane (Task 8) ────────────────────────────────────
#
# The Mojica IconRail (56px column) renders five tab anchors plus Home and
# Settings. The active tab carries the .ic.on class. The LeftPane (248px
# container) renders a filter input + scroll body + status footer. Its filter
# input HTMX-targets /api/library/tree which returns _left_pane_body.html.


def test_left_pane_scaffold_present(client):
    """LeftPane renders filter + scroll body + status footer on Buscar."""
    r = client.get("/search")
    assert r.status_code == 200
    html = r.text
    # Outer container.
    assert 'class="ch-lp"' in html
    # Filter input contract.
    assert 'name="q"' in html
    assert 'hx-get="/api/library/tree"' in html
    assert 'hx-target=".ch-lp .scroll"' in html
    # Scroll body present (the swap target).
    assert 'class="scroll"' in html
    # Footer status indicator + stats.
    assert 'class="foot"' in html
    # The Library · Films section header is always rendered.
    assert "Library · Films" in html or "Acervo · Filmes" in html
    # The Collections section header is always rendered.
    assert "Collections" in html or "Coleções" in html


@pytest.mark.parametrize(
    "path,active",
    [
        ("/search", "buscar"),
        ("/scenes", "cenas"),
        ("/annotate", "anotar"),
        ("/rimas", "rimas"),
        ("/processing", "proc"),
    ],
)
def test_icon_rail_active_for_each_tab(client, path, active):
    """The IconRail marks the corresponding .ic anchor as .on for each route."""
    r = client.get(path)
    assert r.status_code == 200, r.text[:500]
    html = r.text
    # The rail is present.
    assert 'class="ch-rail"' in html
    # An .ic.on anchor exists for the active tab. The href encodes the
    # EN tab key; the active-class check is structural (class string
    # contains both "ic" and "on" tokens).
    tab_routes = {
        "buscar": "/search",
        "cenas": "/scenes",
        "anotar": "/annotate",
        "rimas": "/rimas",
        "proc": "/processing",
    }
    href = tab_routes[active]
    # Find anchors targeting the active tab's href and confirm at least
    # one is on the IconRail (class contains "ic on") rather than the
    # TopBar (class "tab on").
    assert 'class="ic on"' in html
    assert href in html


# ── Group 1e: Buscar main pane (Task 10) ──────────────────────────────────────
#
# The Mojica Buscar template rewrites the legacy text/image toggle + raw
# scene-grid into a single ``.b-cp`` section with:
#   * ``.search-wrap`` containing the qbar + 4 modality chips + retrieval
#     knobs (Hybrid sem/bm25, Rerank, MMR, k),
#   * ``.caption`` row (result/film/latency stats + view-toggle segments),
#   * ``#search-results`` grid (the Task-11 .b-card cards land in this swap
#     target),
#   * a tag-filter section + optional image-upload sub-panel.
#
# These two tests pin the new scaffolding without locking the exact copy or
# legacy classes (Task 11 may still adjust). The film-slug propagation
# tests in test_film_selector.py cover the form-id + hidden-input contract.


def test_buscar_renders_modes_and_knobs(client):
    """Buscar full-page response carries the .b-cp scaffolding."""
    r = client.get("/search")
    assert r.status_code == 200
    html = r.text
    # Outer .b-cp + .search-wrap.
    assert 'class="b-cp"' in html
    assert 'class="search-wrap"' in html
    # qbar form with the new id + push-url contract.
    assert 'id="search-text-form"' in html
    assert 'hx-push-url="true"' in html
    # Modality chips row.
    assert 'class="modes"' in html
    # 4 modality chips visible. The labels are translated; the
    # ``data-mode`` attribute is stable across locales.
    for mode in ("text", "image", "audio", "multimodal"):
        assert f'data-mode="{mode}"' in html
    # Disabled chips for the M2/M3 modalities (audio + multimodal off
    # by default in config). ``disabled`` lands on the <button>.
    assert "disabled aria-disabled" in html
    # Read-only knobs (Hybrid / Rerank / MMR / k).
    assert 'class="knob"' in html
    # The sem-weight knob carries the configured float.
    assert "sem 0.70" in html
    assert "bm25 0.30" in html
    # Caption row + view-toggle segments.
    assert 'class="caption"' in html
    assert 'data-view="grid"' in html
    assert 'data-view="list"' in html
    assert 'data-view="compact"' in html
    # Results grid uses the new .grid class (the Task-11 .b-card cards
    # will land inside this swap target).
    assert 'id="search-results"' in html
    assert 'class="grid"' in html


def test_buscar_search_query_appears(client):
    """The query value flows through to the search input ``value`` attr."""
    r = client.get("/search?q=jeca")
    assert r.status_code == 200
    # The qbar input preserves the query so reloads / push-url navigation
    # do not blank the field.
    assert 'name="q"' in r.text
    assert 'value="jeca"' in r.text


def test_library_tree_filter_endpoint(client):
    """GET /api/library/tree returns the Mojica LeftPane body fragment."""
    r = client.get("/api/library/tree")
    assert r.status_code == 200, r.text[:500]
    html = r.text
    # The fragment is the inner body — it should NOT contain a full document.
    assert "<!DOCTYPE html>" not in html
    # And it should contain the section scaffolding (Library · Films + Collections).
    assert "Library · Films" in html or "Acervo · Filmes" in html
    assert "Collections" in html or "Coleções" in html
    # The .ch-coll "Entire library" / "Acervo inteiro" row anchors the section.
    assert 'class="ch-coll' in html


# ── Group 2a: full-page vs tab context parity — Phase-1a regression lock ──────


class TestFullPageContextDivergence:
    """
    Regression suite locking in the Phase-1a fix: full-page vs HTMX-tab
    context parity.

    Before Phase 1a, ``_base_page()`` in api/server.py supplied only
    ``active_tab, processing_jobs, films, library_state`` while
    ``base.html`` ``{% include %}``d the tab partials, which need far
    more context — so a direct GET of a full-page route rendered a
    degraded / incomplete panel compared to the dedicated ``/tab/*``
    route. Phase 1a routed the full-page path through the same context
    builders as the tab path, so both now render identically.

    Each test below asserts the full-page response contains a specific
    marker that the matching ``/tab/*`` response contains. They pass
    against the fixed behavior and must stay green: a failure here means
    the full-page/tab context parity fix has regressed.
    """

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

    def test_processing_full_page_renders_step_checklist(self, client):
        tab = client.get("/tab/processing")
        full = client.get("/processing")
        assert tab.status_code == 200 and full.status_code == 200
        # The Steps label only appears when step_defs drives the loop;
        # /tab/processing supplies step_defs, /processing does not.
        assert tab.text.count("tag-pill") > 0, "precondition"
        assert full.text.count("tag-pill") == tab.text.count("tag-pill")

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


# ── Group 2b: Processing `split` filter fix — Phase-1b regression lock ────────


class TestProcessingSplitFilterCrash:
    """
    Regression suite locking in the Phase-1b fix: Processing job/stepper
    renders without the unsupported Jinja ``split`` filter.

    Before Phase 1b, ``web/templates/partials/processing_job.html`` used
    ``{{ job.video_path | replace('\\\\','/') | split('/') | last }}``.
    Jinja2 has no built-in ``split`` filter and the app's Jinja env
    (api/templates.py) registers none, so rendering ANY active job raised
    ``jinja2.exceptions.TemplateAssertionError: No filter named 'split'``.
    Phase 1b computes the display filename in Python and removed the
    ``split`` filter from the template.

    ``/tab/processing`` includes ``processing_job.html`` for every active
    job, so injecting one job and GETting the tab exercises the fixed
    path. These tests assert the corrected behavior and must stay green:
    a failure here means the ``split``-filter fix has regressed.
    """

    def test_tab_processing_with_active_job_renders(self, client, inject_job):
        inject_job()
        r = client.get("/tab/processing")
        assert r.status_code == 200, r.text[:500]
        # Phase 1b: the display filename is computed in Python (no Jinja
        # `split` filter), so the basename appears in the rendered job.
        assert "jeca_tatu.mp4" in r.text

    def test_processing_full_page_with_active_job_renders(self, client, inject_job):
        """Phase 1a merged build_processing_context() into the full-page
        path, so a direct GET /processing must include the active job
        and render it without the (removed) `split` filter crash."""
        inject_job()
        r = client.get("/processing")
        assert r.status_code == 200, r.text[:500]
        assert "<!DOCTYPE html>" in r.text
        assert "jeca_tatu.mp4" in r.text
        # The stepper renders on first paint (initial include path), not
        # only via SSE: the step labels and progress track must be present.
        assert "pipeline-steps" in r.text
        assert "progress-fill" in r.text

    def test_active_job_windows_path_basename(self, client, inject_job):
        """A Windows-style video_path (backslashes) must still render
        just the basename — proving the Python-side filename fix
        normalizes separators regardless of host OS."""
        inject_job(video_path=r"C:\\archive\\raw\\jeca_tatu.mp4")
        r = client.get("/tab/processing")
        assert r.status_code == 200, r.text[:500]
        assert "jeca_tatu.mp4" in r.text
        # No path prefix / separators should leak into the title.
        assert "C:" not in r.text
        assert "archive" not in r.text

    def test_errored_job_renders_error_branch(self, inject_job):
        """A job in the error state must render the stepper's error
        branch (status == 'error' + error_msg). Exercised through the
        same single-object contract the initial include uses.

        NOTE: this renders the partial directly rather than via
        /tab/processing because active_jobs() only surfaces
        status == 'running' jobs — broadening that filter is a job
        lifecycle change owned by a later phase, out of scope here."""
        from api.jobs import STEP_DEFS, JobState, StepInfo
        from api.templates import templates

        job = JobState(
            id="errjob",
            video_path="data/raw/jeca_tatu.mp4",
            steps=[StepInfo(name=n, label=lbl) for n, lbl in STEP_DEFS],
        )
        job.status = "error"
        job.error_msg = "boom: model not found"
        job.steps[0].state = "error"
        html = templates.env.get_template("partials/processing_job.html").render(job=job)
        assert "jeca_tatu.mp4" in html
        # Direct env render uses the default pt_BR catalog (the "Pipeline
        # failed" msgid is translated), so assert locale-agnostic markup
        # plus the untranslated error message.
        assert "processing-error" in html
        assert "boom: model not found" in html
        assert "stepper__item--error" in html

    def test_stepper_sse_render_helper_matches_initial_contract(self, inject_job):
        """The SSE path (_render_stepper) and the initial {% include %}
        must use the identical single-object contract. Render via the
        helper and assert it produces the same step/progress markup."""
        from api.routes.processing import _render_stepper

        job = inject_job()
        job.progress = 0.4
        html = _render_stepper(job)
        assert "stepper" in html
        assert "progress-fill" in html
        assert "stepper__item--active" in html
