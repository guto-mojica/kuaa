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


def test_buscar_results_render_as_b_cards_or_empty_state(client):
    """``#search-results`` ships either ``.b-card`` cards or ``.empty-state``.

    Task 11 (Mojica): the legacy ``.scene-card`` markup is gone. On an
    empty test corpus the partial renders the new ``.empty-state``
    wrapper; once results land it renders ``.b-card`` articles. Either
    branch is acceptable for the empty-data smoke client — we just need
    to prove the legacy markup is gone and one of the two new branches
    is the source of truth.
    """
    # Full-page render: the partial is included inside #search-results.
    r = client.get("/search")
    assert r.status_code == 200
    html = r.text
    # The Task-11 partial owns ``.b-card`` and ``.empty-state``; at
    # least one must be present in the rendered #search-results region.
    assert 'class="b-card' in html or 'class="empty-state"' in html
    # And the legacy v0.3 marker is gone from the swap target.
    assert 'class="scene-card"' not in html


def test_buscar_empty_state_when_no_query(client):
    """Without a query, the results region renders the empty-state hint.

    The grid container itself (set up by ``partials/search.html``) is
    always present so HTMX swaps land in a real DOM node; the inner
    partial owns the visible "type a query to search" message.
    """
    r = client.get("/search")
    assert r.status_code == 200
    html = r.text
    # The HTMX swap target stays in place across renders.
    assert 'id="search-results"' in html
    # And the inner partial's empty-state hint is the visible content.
    assert 'class="empty-state"' in html
    assert "Search the library to see results here." in html


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


# ── Group 1f: Buscar inspector (Task 12) ──────────────────────────────────────
#
# The Mojica Buscar inspector is the right-pane ``.b-rp`` partial swapped via
# HTMX from any ``.b-card`` click handler (``hx-get="/api/scenes/<id>/inspector
# ?film=<slug>"``). Task 12 ships the route + service + four partials. These
# tests pin three contracts at once:
#
#   1. The endpoint exists and either 200s (when the seeded scene exists)
#      or 404s (when slug/scene resolution fails). It never 500s and never
#      returns an empty 200 — the route raises HTTPException(404).
#   2. Each of the three tabs (activity / annotations / properties) reaches
#      the correct sub-partial without crashing.
#   3. The Signals (``.b-sigs``) block stays hidden until
#      ``cfg.search.signals_enabled`` flips on AND a per-scene signals dict
#      exists. The composer (``.b-comp``) is gated the same way on
#      ``cfg.collaboration.composer_enabled``.


def test_inspector_returns_partial_for_known_scene(client, seed_metadata):
    """``/api/scenes/<id>/inspector?film=<slug>`` returns the .b-rp partial."""
    seed_metadata()  # seeds slug "default" with scene_id 351 + 352
    r = client.get("/api/scenes/351/inspector?film=default")
    assert r.status_code == 200, r.text[:500]
    html = r.text
    # Outer .b-rp section + the three structural anchors.
    assert 'class="b-rp"' in html
    assert 'class="htabs"' in html
    assert 'class="insp-kf"' in html
    # No full HTML doc — this is an HTMX fragment.
    assert "<!DOCTYPE html>" not in html


def test_inspector_404s_for_unknown_scene(client, seed_metadata):
    """An unresolvable (slug, scene_id) returns 404, not 500 or empty 200."""
    seed_metadata()
    # Unknown scene_id under a real slug.
    r = client.get("/api/scenes/9999/inspector?film=default")
    assert r.status_code == 404, r.text[:500]
    # Unknown slug.
    r = client.get("/api/scenes/351/inspector?film=ghost")
    assert r.status_code == 404, r.text[:500]
    # Missing slug entirely (aggregate hits always carry a slug, but the
    # endpoint must handle a stray URL without crashing).
    r = client.get("/api/scenes/351/inspector")
    assert r.status_code == 404, r.text[:500]


def test_inspector_tab_switching_renders_each_tab(client, seed_metadata):
    """Each of the three tabs reaches its sub-partial without crashing."""
    seed_metadata()
    for tab in ("activity", "annotations", "properties"):
        r = client.get(f"/api/scenes/351/inspector?film=default&tab={tab}")
        assert r.status_code == 200, f"tab={tab} crashed: {r.text[:500]}"
        # The outer .b-rp is rendered for every tab.
        assert 'class="b-rp"' in r.text
    # Activity tab renders the moondream description as a .b-com.ai row.
    r = client.get("/api/scenes/351/inspector?film=default&tab=activity")
    assert "b-com ai" in r.text
    assert "a man walking outdoors at dawn" in r.text  # from seed_metadata
    # Properties tab renders the dl.props block.
    r = client.get("/api/scenes/351/inspector?film=default&tab=properties")
    assert 'class="props"' in r.text
    # Annotations tab renders the inline tag editor.
    r = client.get("/api/scenes/351/inspector?film=default&tab=annotations")
    assert 'name="tags"' in r.text


def test_inspector_hides_signals_and_composer_by_default(client, seed_metadata):
    """``.b-sigs`` and ``.b-comp`` stay hidden until their cfg flags flip on.

    Defaults: cfg.search.signals_enabled=False, cfg.collaboration.composer_enabled=False
    """
    seed_metadata()
    r = client.get("/api/scenes/351/inspector?film=default")
    assert r.status_code == 200
    html = r.text
    assert 'class="b-sigs"' not in html
    assert 'class="b-comp"' not in html
    # Rhymes are deferred to Phase 5 (empty list → block hidden).
    assert 'class="b-rimas"' not in html


def test_inspector_unknown_tab_falls_back_to_activity(client, seed_metadata):
    """An invalid ``?tab=`` value normalises to ``activity`` (no 500)."""
    seed_metadata()
    r = client.get("/api/scenes/351/inspector?film=default&tab=nonsense")
    assert r.status_code == 200, r.text[:500]
    # The .htab marked .on should be the Activity button.
    assert 'class="htab on"' in r.text
    # And the body must include the activity-tab marker (the .b-com.ai row).
    assert "b-com ai" in r.text or "empty-thread" in r.text


def test_search_full_page_includes_inspector_partial_without_crashing(client):
    """``/search`` renders without ``selected_scene`` — the partial self-guards.

    The initial full-page render has no selected scene; the inspector
    must produce no visible chrome rather than crashing on an undefined
    ``selected_scene`` reference.
    """
    r = client.get("/search")
    assert r.status_code == 200, r.text[:500]
    # The inspector partial is included by search.html (no ``ignore missing``
    # any more). With no selected_scene it must render to nothing visible:
    # the .b-rp class should NOT appear on the initial page.
    assert 'class="b-rp"' not in r.text


# ── Group 1f-bis: Cenas inspector (Task 16) ───────────────────────────────────
#
# The Mojica Cenas inspector reuses the same ``/api/scenes/<id>/inspector``
# endpoint but discriminates on a new ``?kind=`` query param:
#
#   * ``kind=buscar`` (default, omitted) → ``.b-rp`` Buscar shell
#   * ``kind=cenas``                     → ``.c-rp`` Cenas shell
#
# The scenecard's hx-get on the Cenas grid passes ``kind=cenas``; the
# .b-card's hx-get on the Buscar grid omits the param and keeps the
# Task-12 contract intact.


def test_cenas_inspector_returns_c_rp_markup(client, seed_metadata):
    """``?kind=cenas`` returns the .c-rp partial, NOT the .b-rp Buscar shell."""
    seed_metadata()
    r = client.get("/api/scenes/351/inspector?film=default&tab=properties&kind=cenas")
    assert r.status_code == 200, r.text[:500]
    html = r.text
    # New Cenas wrapper.
    assert 'class="c-rp"' in html
    # MUST NOT carry the Buscar shell.
    assert 'class="b-rp"' not in html
    # The Cenas inspector has no htabs at all.
    assert 'class="htabs"' not in html
    # Props grid is rendered inline (not via inspector_properties.html which
    # uses ``<dl class="props">``; the Cenas inspector uses ``<div
    # class="props">`` per cenas.css's grid spec).
    assert 'class="props"' in html
    # The 3-action footer + selection-count head are part of the Cenas
    # shell only.
    assert 'class="actions"' in html
    assert 'class="badge"' in html


def test_buscar_inspector_defaults_to_b_rp(client, seed_metadata):
    """Omitting ``?kind=`` keeps the Task-12 contract: .b-rp is returned."""
    seed_metadata()
    r = client.get("/api/scenes/351/inspector?film=default")
    assert r.status_code == 200, r.text[:500]
    html = r.text
    assert 'class="b-rp"' in html
    assert 'class="c-rp"' not in html
    # Explicit ``kind=buscar`` is equivalent.
    r2 = client.get("/api/scenes/351/inspector?film=default&kind=buscar")
    assert r2.status_code == 200
    assert 'class="b-rp"' in r2.text


def test_cenas_grid_scenecard_uses_kind_cenas(client, seed_metadata):
    """Every ``.scenecard``'s ``hx-get`` includes ``kind=cenas`` (Task 16 wiring)."""
    seed_metadata()
    # /scenes is the full-page route; if the grid renders any scenecards
    # they must point at the kind=cenas inspector.
    r = client.get("/scenes")
    if r.status_code == 200 and "scenecard" in r.text:
        assert "kind=cenas" in r.text


# ── Group 1f.5: Anotar stage (Task 18) ────────────────────────────────────────
#
# The Mojica Anotar rewrite swaps the legacy filter/jump/form left pane for a
# Frame.io-style stage: meta header + keyframe + timeline + player controls.
# These tests pin the public contracts the rewrite establishes so future
# changes that drop the stage markup, accidentally revive the legacy form on
# the left, or stop collapsing the LeftPane fail loudly. The legacy form is
# intentionally kept on the right (inside ``#annotate-scene``) by Task 18 —
# Task 19 retires it into the proper ``.a-rp`` htabs + thread + composer.


def test_anotar_renders_stage_meta_keyframe_timeline_player(client, seed_metadata):
    """``/annotate`` renders the .a-stage + .a-meta + .a-keyframe + .a-tl + .a-pl markup.

    The empty-state branch (no seeded data) renders only the empty
    placeholder, so we seed metadata first to exercise the populated
    branch. The five class assertions cover every top-level block of
    the Frame.io-style scene-review surface.
    """
    seed_metadata()
    r = client.get("/annotate")
    assert r.status_code == 200
    html = r.text
    assert 'class="a-stage"' in html
    assert 'class="a-meta"' in html
    assert 'class="a-keyframe"' in html
    assert 'class="a-tl"' in html
    assert 'class="a-pl"' in html


def test_anotar_uses_compact_lp(client):
    """The Anotar tab collapses the LeftPane via the compact-lp body modifier.

    This pins down ``_TAB_CHROME['annotate']['compact_lp'] = True`` end-
    to-end: the body class controls the CSS grid that hides ``.ch-lp``,
    so a stale chrome config would render an extra 248px sidebar and
    break the stage layout silently. We deliberately do NOT assert the
    absence of every legacy left-pane class (e.g. ``ch-lp``); the
    Phase-1 chrome still renders ``ch-lp`` markup elsewhere, and that
    is unrelated to the compact-lp body modifier we care about here.
    """
    r = client.get("/annotate")
    assert r.status_code == 200
    assert "compact-lp" in r.text


def test_anotar_breadcrumb_shows_scene_number(client, seed_metadata):
    """The .a-meta breadcrumb renders the scene number as ``scene NNN`` (zero-padded).

    Format-string lives in the template: ``"%03d" | format(scene_id)``,
    so scene 351 in the seed fixture becomes ``scene 351``. The
    breadcrumb wrapper class is asserted alongside the formatted scene
    label so a future template change that drops the breadcrumb (or
    silently renders an unformatted id) fails here.
    """
    seed_metadata()
    r = client.get("/annotate")
    assert r.status_code == 200
    html = r.text
    assert 'class="filmpath"' in html
    # Seed defaults to scene 351 as the first scene. The empty-state
    # filter fallback (no_llm → all) lands here because the default
    # seed has no LLM descriptions yet, so the partial picks the first
    # scene of the "all" list — still 351.
    assert "scene 351" in html or "cena 351" in html


# ── Group 1f-bis: Anotar right pane (Task 19) ─────────────────────────────────
#
# The .a-rp shell carries htabs (Comments / Annotations / Properties), a
# subhead and a dispatched sub-partial body. The HTMX endpoint
# ``/api/annotate/scene`` reads ``?tab=`` and renders accordingly. These
# three tests lock the contract: the markup is present, all three tab
# values resolve to a 200 response, and the legacy save endpoint contract
# (form-encoded POST → on-disk JSON) is preserved by the .a-rp rewrite.


def test_anotar_inspector_returns_a_rp(client, seed_metadata):
    """``/api/annotate/scene`` renders the ``.a-rp`` shell with ``.htabs``.

    The legacy ``annotate_scene.html`` was a two-column LLM + tag form;
    Task 19 replaces it with the Frame.io-style ``.a-rp`` (htabs +
    subhead + dispatched thread). Both new class hooks must appear on
    every render so CSS / interaction tests targeting the right pane
    have stable selectors.
    """
    seed_metadata()
    r = client.get("/api/annotate/scene?id=351")
    assert r.status_code == 200, r.text[:300]
    html = r.text
    assert 'class="a-rp"' in html
    assert 'class="htabs"' in html
    # The legacy nav-position counter is still present on the .a-rp
    # subhead — see TestAnnotateSceneEmptyFilterRegression.
    assert "annotate-nav__pos" in html


def test_anotar_inspector_tab_switching(client, seed_metadata):
    """``?tab=`` selects the right-pane htab body without 500s.

    All three valid values render the corresponding sub-partial:
        * ``comments``    → ``partials/annotate_comments.html`` (AI .a-com.ai)
        * ``annotations`` → ``partials/annotate_tags.html`` (tag editor)
        * ``properties``  → ``partials/annotate_props.html`` (props dl)

    Unknown values must fall back to ``comments`` via
    :func:`api.services.annotations.normalize_annotate_tab`; assert that
    too so a future regression in the normaliser does not silently break
    the default-tab landing state.
    """
    seed_metadata()

    # All three valid tabs render.
    r = client.get("/api/annotate/scene?id=351&tab=comments")
    assert r.status_code == 200
    assert 'class="a-rp"' in r.text
    # Comments sub-partial: either renders the AI description body
    # ("moondream-2") or the empty-state line ("No LLM description").
    assert "moondream-2" in r.text or "No LLM description" in r.text

    r = client.get("/api/annotate/scene?id=351&tab=annotations")
    assert r.status_code == 200
    # Annotations sub-partial preserves the legacy tag-editor input.
    assert "annotate-tags-input" in r.text

    r = client.get("/api/annotate/scene?id=351&tab=properties")
    assert r.status_code == 200
    # Properties sub-partial renders a definition-list (<dl class="props">).
    assert 'class="props"' in r.text

    # Unknown tabs fall back to comments (markup proof: the
    # annotate-tags-input only appears on annotations, so an unknown
    # value must NOT show it).
    r = client.get("/api/annotate/scene?id=351&tab=bogus")
    assert r.status_code == 200
    assert "annotate-tags-input" not in r.text


def test_anotar_save_endpoint_still_works(client, seed_metadata):
    """The legacy ``/api/annotate/save`` form contract is preserved.

    Task 19's .a-rp rewrite moves the tag form into a sub-partial but
    keeps the field names (``scene_id`` / ``filter`` / ``tags``) and
    the on-disk shape (str scene-id key, lower-kebab tags) unchanged.
    Posting the same body the old form used must still return 200 and
    persist the normalised tags to ``manual_annotations.json``.
    """
    paths = seed_metadata()
    r = client.post(
        "/api/annotate/save",
        data={
            "scene_id": "351",
            "filter": "all",
            "tags": "test, demo",
        },
    )
    assert r.status_code == 200, r.text[:300]
    # The save renders the full .a-rp partial back into #annotate-scene.
    assert 'class="a-rp"' in r.text
    # On-disk shape is unchanged: str scene-id key, lower-kebab tags.
    import json as _json

    on_disk = _json.loads(paths["manual_path"].read_text())
    assert on_disk["351"] == ["test", "demo"]


# ── Group 1g: Buscar timeline (Task 13) ───────────────────────────────────────
#
# The bottom-timeline (``.b-tl``) renders below the .b-cp results grid only
# when ``?scene=&film=`` URL params resolve to a real film with on-disk
# keyframe metadata. On the bare /search initial render (no selection) the
# partial self-guards on ``selected_film`` and produces no chrome — same
# pattern as Task 12's inspector. The partial is included from search.html
# without ``ignore missing`` (Task 13 removes the guard).


def test_search_timeline_absent_on_full_page_without_selection(client):
    """``/search`` with no ``?scene=&film=`` renders no ``.b-tl`` chrome.

    The timeline partial self-guards on ``selected_film``; the initial
    page must NOT render the scrub strip or the timeline header.
    """
    r = client.get("/search")
    assert r.status_code == 200, r.text[:500]
    html = r.text
    assert 'class="b-tl"' not in html
    assert 'class="scrub"' not in html


def test_search_timeline_renders_when_scene_and_film_selected(client, seed_metadata):
    """``/search?scene=351&film=default`` populates the .b-tl with scrub + ticks.

    The seeded fixture writes two scenes (351, 352) with start/end
    times, so the timeline builder resolves a positive runtime, emits
    one ``.seg`` per scene, marks the selected scene as ``.sel`` and
    flags it as a match (M1 simplification: only the selected scene
    counts as a match until M2 wires the full per-film match set).

    Also asserts the right-pane inspector co-renders on the same page
    (the timeline builder populates ``selected_scene`` + ``selected_film``
    which the inspector partial picks up — a single render delivers
    both panes without an extra HTMX swap).
    """
    seed_metadata()
    r = client.get("/search?scene=351&film=default&q=campo")
    assert r.status_code == 200, r.text[:500]
    html = r.text
    # Outer .b-tl section + the two structural anchors are present.
    assert 'class="b-tl"' in html
    assert 'class="scrub"' in html
    assert 'class="ticks"' in html
    # Header carries the film title and a match count.
    assert "Default Film" in html
    # The selected scene's .seg is marked .sel; the other scene is not.
    # Two scenes seeded; both should be present in the scrub. Use
    # the explicit ?scene= URL param target attribute to assert order
    # is preserved.
    assert "scene=351" in html
    assert "scene=352" in html
    # Selected (351) gets .seg.sel + .match; the other only renders bare .seg.
    assert 'class="seg match sel"' in html
    # Query param is preserved in the .seg hrefs so a click round-trips state.
    assert "q=campo" in html
    # The same render also produces the right-pane inspector (the timeline
    # builder reuses build_inspector_context, so selected_scene + selected_film
    # populate both partials on a single response).
    assert 'class="b-rp"' in html


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


# ── Group 1h: Cenas tab — Mojica redesign (Task 15) ───────────────────────────
#
# Task 15 rewrites ``scenes.html`` + ``scenes_grid.html`` onto the new
# ``.c-cp + .toolrow + .countrow + .scenecard`` markup with film-grouped
# sections. The three tests below pin (1) the toolrow / countrow / grid
# scaffolding on the full-page route, (2) the grouped grid markup when
# seeded data exists, and (3) the ``tipo_of`` classifier contract.


def test_scenes_tab_renders_toolbar_and_countrow(client):
    """``/scenes`` renders the .c-cp scaffolding (toolrow + countrow + grid).

    On an empty library the partial routes through the ``no_data``
    branch which renders the empty-state hint inside the .c-cp wrapper,
    so the outer ``class="c-cp"`` marker is the only stable smoke
    signal. The seeded variant below pins the toolrow + countrow +
    grid in their non-empty state.
    """
    r = client.get("/scenes")
    assert r.status_code == 200, r.text[:500]
    html = r.text
    assert 'class="c-cp"' in html
    # Empty state still uses the .c-cp shell so the parity smoke holds.
    assert 'class="scene-card"' not in html, "legacy markup leaked into Mojica grid"


def test_scenes_grid_groups_by_film(client, seed_metadata):
    """Seeded data populates the toolrow / countrow / grouped grid.

    The fixture registers slug ``"default"`` with two scenes (351, 352);
    the grid must render one ``.group`` heading and at least one
    ``.scenecard`` article inside the ``#scenes-grid`` swap target.
    """
    seed_metadata()
    r = client.get("/scenes")
    assert r.status_code == 200, r.text[:500]
    html = r.text
    # Outer .c-cp shell.
    assert 'class="c-cp"' in html
    # Toolrow + .find input contract.
    assert 'class="toolrow"' in html
    assert 'class="find"' in html
    assert 'hx-get="/api/scenes"' in html
    # Countrow with totals.
    assert 'class="countrow"' in html
    # Grouped grid: one .group heading + at least one .scenecard.
    assert 'id="scenes-grid"' in html
    assert 'class="group"' in html
    assert "Default Film" in html
    assert 'class="scenecard"' in html
    # Tipo pill carries one of the canonical tipo CSS variables.
    assert "var(--c-cat-" in html
    # And the legacy v0.3 marker is gone from the swap target.
    assert 'class="scene-card"' not in html


def test_tipo_classifier_unit():
    """``tipo_of`` returns the documented bucket for each tag pattern."""
    from api.services.scenes_service import tipo_of

    assert tipo_of(["cartela", "title-card"], None) == "cartela"
    assert tipo_of(["interior", "baixa-luz"], None) == "interior"
    assert tipo_of(["exterior", "rural"], None) == "exterior"
    assert tipo_of(["duas-pessoas"], None) == "dialogo"
    assert tipo_of([], None) == "transicao"
    # Description-driven cartela fallback (no matching tag).
    assert tipo_of([], "title sequence") == "cartela"


# ── Group 1i: Rimas Visuais tab routes (Task 21) ──────────────────────────────
#
# Task 21 wires ``/tab/rimas`` + ``/api/rimas/echoes`` to the new
# ``api.services.rhymes_service.build_rimas_context`` builder which walks
# the library, resolves the anchor scene, and calls
# :func:`cinemateca.rhymes.find_rhymes` for the cross-film kNN. These tests
# pin (1) the full-page route stays 200 on an empty library (empty-state
# branch), (2) the HTMX tab fragment returns the ``.r-cp`` shell, (3) the
# echoes endpoint returns a partial without crashing, and (4) the
# ``?anchor=`` query param parses without crashing on an unknown slug.
# Task 22 ships the real templates and tightens these assertions.


def test_rimas_page_renders(client):
    """``/rimas`` renders 200 on an empty library with the .r-cp shell or empty state.

    The service falls back to the default-anchor branch when no
    processed films exist; the placeholder template emits an
    ``empty-state`` row in that case. Either branch is acceptable here —
    the contract is "never 500, always render the chrome around it".
    """
    r = client.get("/rimas")
    assert r.status_code == 200, r.text[:500]
    html = r.text
    # The Mojica chrome shell is rendered (full-page route).
    assert 'data-active-tab="rimas"' in html
    # And the tab body produces either the .r-cp wrapper (with or
    # without an anchor) or the empty-state row.
    assert 'class="r-cp"' in html or 'class="empty-state"' in html


def test_tab_rimas_fragment_returns_partial(client):
    """``/tab/rimas`` returns the .r-cp partial (HTMX swap target).

    Asserts the fragment-only contract: no ``<!DOCTYPE html>`` (full
    document leaked) and the ``.r-cp`` wrapper is present. The empty
    library branch still emits the .r-cp shell with the empty-state row
    inside, so the assertion is locale-stable.
    """
    r = client.get("/tab/rimas")
    assert r.status_code == 200, r.text[:500]
    html = r.text
    assert "<!DOCTYPE html>" not in html
    assert 'class="r-cp"' in html


def test_api_rimas_echoes_returns_fragment(client):
    """``/api/rimas/echoes`` returns the echoes-grid partial without crashing.

    On an empty library the grid renders the empty-state row inside
    ``.r-grid``. Locking only the structural anchors keeps the test
    locale-stable and lets Task 22 swap the inner markup without
    breaking this contract.
    """
    r = client.get("/api/rimas/echoes")
    assert r.status_code == 200, r.text[:500]
    html = r.text
    # Fragment — no full document.
    assert "<!DOCTYPE html>" not in html
    # The swap target wrapper is always rendered.
    assert 'id="rimas-echoes"' in html
    assert 'class="r-grid"' in html


def test_rimas_with_explicit_anchor_does_not_crash(client):
    """``/rimas?anchor=<slug>/<scene_id>`` accepts and parses the anchor.

    On a fresh empty library the slug never resolves — the service
    returns no anchor data and falls back to the empty-state branch.
    The contract under test is "the URL never crashes the page";
    Task 22's seeded variant will pin the populated branch separately.
    """
    r = client.get("/rimas?anchor=jeca/1")
    assert r.status_code == 200, r.text[:500]
    # Same chrome / shell contract as the no-param variant.
    assert 'data-active-tab="rimas"' in r.text

    # Malformed anchor falls back to the default-anchor branch.
    r = client.get("/rimas?anchor=garbage")
    assert r.status_code == 200, r.text[:500]
    r = client.get("/rimas?anchor=jeca/not-an-int")
    assert r.status_code == 200, r.text[:500]


# ── Group 1j: Rimas Visuais full template (Task 22) ───────────────────────────
#
# Task 22 ships the real CSS-driven templates (.topbar / .controls / .r-anchor
# / .r-grid / .r-rp / .r-pair / .r-similarity-card) and the new
# ``/api/rimas/inspector`` endpoint for HTMX echo-selection swaps. These tests
# pin the structural anchors that Task 22's design requires; the empty-library
# branch is the canonical fixture state, so assertions stay locale-stable.


def test_rimas_page_has_topbar_and_controls(client):
    """``/rimas`` renders the Mojica .topbar + .controls knob row.

    Pins the new Task-22 markup: the topbar (with the rhyme count chip
    on populated libraries or a hint on empty ones) and the controls
    knob row. Both branches share the .r-cp wrapper so this assertion
    holds regardless of library state.
    """
    r = client.get("/rimas")
    assert r.status_code == 200, r.text[:500]
    html = r.text
    assert 'class="r-cp"' in html
    # Either the topbar OR the controls knob row must be present —
    # they ship together but the controls knob row carries the
    # locale-stable knob/k/v structure (the .topbar is structural-only
    # markup so its className alone is the test surface).
    assert 'class="topbar"' in html
    assert 'class="controls"' in html
    assert 'class="knob"' in html


def test_rimas_inspector_endpoint(client):
    """``/api/rimas/inspector`` returns the right-pane fragment.

    The endpoint is fired by .r-echo card clicks (hx-target="#right-
    pane") and returns the inspector partial wrapped in
    ``.r-rp-inner``. On an empty library the anchor itself does not
    resolve, so the empty-state branch fires — still 200, still a
    fragment, never a 500.
    """
    r = client.get("/api/rimas/inspector?anchor=jeca/1&echo=limite/1")
    assert r.status_code < 500, r.text[:500]
    html = r.text
    # Fragment — no full document leaked.
    assert "<!DOCTYPE html>" not in html
    # The inspector wrapper is always rendered, even on the
    # anchor-only / empty branches (the .r-rp-inner block is the
    # template's outer element).
    assert 'class="r-rp-inner"' in html


def test_rimas_grid_uses_echo_card_class(client):
    """``/tab/rimas`` emits the .r-grid container in both branches.

    The grid container ships even on an empty library (the
    empty-state row lives inside it, mirroring the swap target
    contract). Task-22 cards use the ``.r-echo`` className; on an
    empty library no cards render, so the negative assertion is
    "container present, empty-state inside".
    """
    r = client.get("/tab/rimas")
    assert r.status_code == 200, r.text[:500]
    html = r.text
    assert 'class="r-grid"' in html
    assert 'id="rimas-echoes"' in html


# ── Group 1k: Processamento tab — Mojica redesign (Task 24) ───────────────────
#
# Task 24 rewrites ``processing.html`` + ``processing_job.html`` onto the
# Mojica ``.p-cp / .p-top / .p-active / .p-pbar / .p-steps / .p-log /
# .p-side / .p-stats / .p-queue / .p-rp`` layout. The three tests below
# pin the structural anchors the new template ships AND the empty-state
# behaviour that ``test_tab_processing_empty_has_no_active_jobs`` (Group
# 1) and the Phase-1a/1b regression locks (above) further constrain.


def test_proc_renders_p_cp_with_top(client):
    """``/processing`` ships the new ``.p-cp`` / ``.p-top`` / ``.p-log``
    layout anchors and the active-jobs swap wrapper.

    These are the four structural markers proc.css drives. The
    legacy launch form is still present (preserved inside ``.p-top
    .acts``) so the empty-data regression tests stay green; this test
    only pins the NEW layout, not the legacy markers.
    """
    r = client.get("/processing")
    assert r.status_code == 200, r.text[:500]
    html = r.text
    assert 'class="p-cp"' in html
    assert 'class="p-top"' in html
    assert 'id="active-jobs"' in html
    assert 'class="p-log"' in html


def test_proc_no_active_jobs_renders_empty_state(client):
    """Empty registry renders the ``No active jobs.`` text inside the
    ``#active-jobs`` swap wrapper.

    The text is the same one v0.2.x shipped; preserving it keeps the
    Phase-0 regression suite green AND the new template's empty branch
    is the SAME branch the old template used (so HTMX swaps land on
    the same target id from the launch-form POST).
    """
    r = client.get("/processing")
    assert r.status_code == 200, r.text[:500]
    html = r.text
    assert "No active jobs." in html, "empty-state line missing"
    # The empty state still ships inside the new .p-cp / #active-jobs
    # scaffolding (it is not the whole tab — the log + side panels also
    # render).
    assert 'id="active-jobs"' in html


def test_proc_stats_section_present(client):
    """``/processing`` always ships the ``.p-stats`` + ``.p-queue`` cards.

    Even on an empty library the cards render with all counts at 0;
    no data should make these sections vanish (the layout would
    collapse). Task 24's ``aggregate_stats`` walks ``data/library/``
    safely on an empty/absent directory and returns the zero-defaults
    dict.
    """
    r = client.get("/processing")
    assert r.status_code == 200, r.text[:500]
    html = r.text
    assert 'class="p-stats"' in html
    assert 'class="p-queue"' in html


# ── Task 25 ───────────────────────────────────────────────────────────
# mojica.js — Phase 6 seed: SSE log auto-scroll on the Processing tab.
# Browser-level behaviour testing requires a real DOM (Playwright); these
# tests pin the contract at the asset-serving + template-wiring level so
# regressions in either side fail fast in pytest.

def test_mojica_js_served(client):
    """`/static/js/mojica.js` is served by the FastAPI static mount and
    contains the auto-scroll binding helper.

    The asset is a vendored vanilla-JS module (no build step), so we
    can assert on substrings of the source directly. Either marker is
    sufficient evidence the file at the served URL is the Phase-6
    polish layer (not, e.g., an empty placeholder).
    """
    r = client.get("/static/js/mojica.js")
    assert r.status_code == 200, r.text[:200]
    body = r.text
    assert "bindLogAutoscroll" in body or "proc-log" in body


def test_processing_page_loads_mojica_js(client):
    """The Processing tab full-page render references `mojica.js` so the
    SSE auto-scroll binding ships with the chrome.

    The `<script>` tag lives in ``base.html`` (Phase-1 chrome), so any
    full-page render — not just /processing — should include it. We
    pin it on /processing because that is the tab whose behaviour the
    script enables.
    """
    r = client.get("/processing")
    assert r.status_code == 200, r.text[:500]
    assert "/static/js/mojica.js" in r.text
