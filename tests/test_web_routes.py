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

import re

import pytest

# ── Group 1: empty-data smoke tests (must PASS) ───────────────────────────────

FULL_PAGES = ["/", "/search", "/scenes", "/annotate", "/processing"]
TAB_FRAGMENTS = ["/tab/search", "/tab/scenes", "/tab/annotate", "/tab/processing"]


@pytest.mark.parametrize("path", FULL_PAGES)
def test_full_page_routes_respond(client, path):
    """Full-page routes return 200 with the new Mojica shell rendered.

    Phase-2 deleted the legacy ``<div class="tab-content" id="tab-content">``
    wrapper; tab partials now drop straight inside ``<main class="ch-main">``
    via their own ``.tab-panel`` div. ``id="ch-main"`` is the post-Phase-2
    stable mount point.
    """
    r = client.get(path)
    assert r.status_code == 200, r.text[:500]
    assert "<!DOCTYPE html>" in r.text
    assert 'id="ch-main"' in r.text


@pytest.mark.parametrize("path", TAB_FRAGMENTS)
def test_tab_fragment_routes_respond(client, path):
    """`/tab/*` partials return 200 and a tab panel (no full HTML doc)."""
    r = client.get(path)
    assert r.status_code == 200, r.text[:500]
    assert 'class="tab-panel"' in r.text
    assert "<!DOCTYPE html>" not in r.text


def test_about_modal_responds(client):
    """/api/about renders the redesigned About modal partial (Task 29).

    The 0.3.0 version assertion the pre-Mojica test carried is replaced
    by a structural check: the modal now reads the runtime
    ``cinemateca.__version__``, and the surface is identified by its
    ``.ab-modal`` panel rather than a literal version string.
    """
    r = client.get("/api/about")
    assert r.status_code == 200, r.text[:500]
    assert 'class="ab-modal"' in r.text


def test_about_modal_renders_model_attributions(client):
    """Every model in the project pipeline gets its own attribution card."""
    r = client.get("/api/about")
    assert r.status_code == 200, r.text[:500]
    # Four attribution cards present (one per pipeline model).
    assert r.text.count('class="ab-model"') == 4
    # Identifiable model names appear (covers both name and badge text).
    body = r.text.lower()
    assert "moondream" in body
    assert "clip" in body
    assert "yolov8" in body
    assert "mtcnn" in body


def test_about_modal_stats_grid(client):
    """The header stats strip renders 4 cells with films/scenes/runtime/years."""
    r = client.get("/api/about")
    assert r.status_code == 200, r.text[:500]
    assert 'class="ab-stats"' in r.text
    # Exactly four stat cells (films, scenes, runtime, year range).
    assert r.text.count('class="ab-stat"') == 4


def test_about_page_renders_full_page(client):
    """/about wraps the modal partial in a standalone page for JS-off users."""
    r = client.get("/about")
    assert r.status_code == 200, r.text[:500]
    assert "<!DOCTYPE html>" in r.text
    # The full-page route shares the modal partial: ``.ab-app`` root +
    # ``.ab-modal`` panel both appear.
    assert 'class="ab-app"' in r.text
    assert 'class="ab-modal"' in r.text


# ── U8: keyboard-shortcut discoverability in About ───────────────────────────
#
# The ``?`` overlay (partials/_help_overlay.html) is unreachable on the
# standalone /about page (it deliberately omits mojica.js). U8 surfaces the
# shortcut map as a STATIC reference inside the shared about_modal partial, so
# both the modal and the standalone page document the shortcuts regardless of
# scripting. These tests pin that the legend renders with its keys + labels.


@pytest.mark.parametrize("path", ["/api/about", "/about"])
def test_about_surfaces_keyboard_shortcuts(client, path):
    """Both About surfaces render the static keyboard-shortcuts reference.

    The ``/about`` standalone page has no mojica.js, so the interactive ``?``
    overlay cannot open there — the static ``.ab-keys`` list is the only way
    the shortcut map is discoverable. It must render on BOTH the HTMX modal
    (``/api/about``) and the full page (``/about``).
    """
    r = client.get(path)
    assert r.status_code == 200, r.text[:500]
    html = r.text
    # The static legend container + its section heading.
    assert 'class="ab-keys"' in html
    assert ">Keyboard shortcuts<" in html
    # Representative shortcut labels (mirrors _help_overlay's nav + universal).
    assert ">Command palette<" in html
    assert ">This help<" in html
    # The keys render as <kbd> chips, not bare text.
    assert "<kbd>1</kbd>" in html
    assert "<kbd>?</kbd>" in html
    assert "<kbd>⌘</kbd>" in html


# ── U11: offline status indicator ────────────────────────────────────────────
#
# A topbar badge reflecting ``navigator.onLine`` reinforces the offline-first
# value prop. It is server-rendered hidden and toggled by a vanilla
# online/offline listener in mojica.js. These tests pin the markup (present in
# the chrome shell, correct aria treatment) and the JS listener contract.


def test_offline_badge_present_in_chrome_with_aria_live(client):
    """The offline badge ships in the chrome shell with a polite live region.

    a11y: the badge conveys a *status* the user should be told about, so it is
    an ``aria-live="polite"`` ``role="status"`` region (NOT aria-hidden — that
    would suppress the very change we want announced). It is server-rendered
    ``hidden`` (online is the default state); the JS toggles the attribute.
    """
    r = client.get("/search")
    assert r.status_code == 200
    html = r.text
    assert 'id="offline-badge"' in html
    assert 'role="status"' in html
    assert 'aria-live="polite"' in html
    # Hidden at rest (online) — the JS removes this on the offline event.
    assert "offline-badge" in html
    badge_start = html.index('id="offline-badge"')
    badge_tag = html[badge_start - 40 : badge_start + 120]
    assert "hidden" in badge_tag, badge_tag
    # The visible short label + the SR sentence both render.
    assert ">Offline<" in html
    assert "Working offline" in html


@pytest.mark.parametrize("path", ["/search", "/scenes", "/annotate", "/rimas"])
def test_offline_badge_on_every_chrome_tab(client, path):
    """The badge lives in base.html chrome, so every full-page tab carries it."""
    r = client.get(path)
    assert r.status_code == 200
    assert 'id="offline-badge"' in r.text


def test_mojica_js_has_offline_listener(client):
    """``mojica.js`` wires the vanilla online/offline listeners to the badge.

    Pins the client contract: the file binds BOTH the ``online`` and
    ``offline`` window events and targets ``offline-badge``. Without these the
    badge would render hidden forever (server state) and never react.
    """
    body = client.get("/static/js/mojica.js").text
    assert "addEventListener('online'" in body
    assert "addEventListener('offline'" in body
    assert "offline-badge" in body
    # Toggling visibility goes through the ``hidden`` attribute (not a class),
    # matching the server-rendered initial state.
    assert "navigator.onLine" in body


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
    """Buscar full-page response carries TopBar/LeftPane/right markers."""
    r = client.get("/search")
    assert r.status_code == 200
    html = r.text
    # TopBar present + brand name.
    assert 'class="ch-top"' in html
    assert "Mojica" in html
    # The icon rail was removed — the top tabs are the sole nav surface.
    assert 'class="ch-rail"' not in html
    # LeftPane (not compact on Buscar).
    assert 'class="ch-lp"' in html
    # Buscar's right pane lives inside .tab-panel (not ch-right), so no
    # with-right grid modifier; the inspector container is present though.
    assert 'id="right-pane"' in html
    # Active tab tag uses the PT slug.
    assert 'data-active-tab="buscar"' in html


def test_base_shell_compact_for_anotar(client):
    """Anotar collapses the left pane via the compact-lp body modifier."""
    r = client.get("/annotate")
    assert r.status_code == 200
    assert "compact-lp" in r.text


def test_base_shell_loads_mojica_js_before_alpine(client):
    """mojica.js MUST execute before alpine.min.js — see base.html comment.

    Alpine 3 auto-starts when ``document.readyState !== 'loading'``,
    which is always true for deferred scripts. ``start()`` dispatches
    ``alpine:init`` synchronously inside alpine.min.js's execution.
    If mojica.js loads AFTER alpine, every
    ``addEventListener('alpine:init', …)`` in our IIFEs misses the
    event, and the entire Alpine-store layer (toasts, help, palette,
    tagFilter, cenasAppearance, cenasFields, buscarView) stays
    unregistered. ``$store.buscarView.mode`` then throws on first
    paint and the seg buttons in Buscar lose their reactive
    highlight. This test pins the script order so that bug can't
    silently regress.
    """
    r = client.get("/search")
    assert r.status_code == 200
    html = r.text
    mojica_pos = html.find("js/mojica.js")
    alpine_pos = html.find("js/alpine.min.js")
    assert mojica_pos != -1, "mojica.js <script> tag missing from base.html"
    assert alpine_pos != -1, "alpine.min.js <script> tag missing from base.html"
    assert mojica_pos < alpine_pos, (
        "mojica.js must come before alpine.min.js so its alpine:init "
        "listeners register before Alpine auto-starts. Swap them back "
        "and the new Buscar view-toggle / Cenas Appearance / Fields "
        "stores all break silently."
    )


def test_eval_shell_loads_mojica_eval_before_alpine(client, monkeypatch):
    """Eval shell pins the same script order — see base shell counterpart.

    ``eval/layout.html`` is a separate shell (does NOT extend base.html),
    so the same Alpine-3 auto-start bug bites it independently. If
    alpine.min.js fires ``alpine:init`` before eval.js's
    ``Alpine.data('evalApp', …)`` registers, the body's
    ``x-data="evalApp(…)"`` throws "evalApp is not defined" and the
    entire grading workspace — keyboard router, grade-button @click,
    blind/compare toggles, row cursor, toasts — goes inert. The pin
    keeps both mojica.js and eval.js executing before alpine.min.js.
    """
    monkeypatch.setenv("EVAL_ADMIN_TOKEN", "test-token")
    r = client.get("/eval?token=test-token")
    assert r.status_code == 200, r.text
    html = r.text
    mojica_pos = html.find("js/mojica.js")
    eval_pos = html.find("js/eval.js")
    alpine_pos = html.find("js/alpine.min.js")
    assert mojica_pos != -1, "mojica.js <script> tag missing from eval/layout.html"
    assert eval_pos != -1, "eval.js <script> tag missing from eval/layout.html"
    assert alpine_pos != -1, "alpine.min.js <script> tag missing from eval/layout.html"
    assert mojica_pos < alpine_pos, (
        "mojica.js must come before alpine.min.js in the eval shell "
        "for the same reason as base.html — see the layout.html comment."
    )
    assert eval_pos < alpine_pos, (
        "eval.js must come before alpine.min.js so Alpine.data('evalApp', …) "
        "is registered before Alpine walks the DOM. Swap them back and the "
        "entire eval grading UI goes inert ('evalApp is not defined')."
    )


def test_base_shell_includes_palette_and_help_roots(client):
    """Polish-layer mount points exist on the index page.

    Task 27 replaced the ``#palette-root`` placeholder with the real
    server-rendered command-palette scaffold (``id="palette"`` + nested
    ``#cp-input`` / ``#cp-list``). Task 28 replaced ``#help-root`` with
    the keyboard-help overlay scaffold (``id="help"`` + ``#kh-title``).
    The toast root stays as an empty mount because ToastBus creates
    toasts imperatively.
    """
    r = client.get("/")
    html = r.text
    assert 'id="palette"' in html
    assert 'id="cp-input"' in html
    assert 'id="help"' in html
    assert 'id="kh-title"' in html
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
# The Mojica IconRail (56px column) renders five tab anchors plus Home.
# The active tab carries the .ic.on class. The LeftPane (248px
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
def test_topbar_active_for_each_tab(client, path, active):
    """The TopBar marks the corresponding tab anchor as .on for each route."""
    r = client.get(path)
    assert r.status_code == 200, r.text[:500]
    html = r.text
    # The icon rail was removed; the top tabs are the sole nav surface.
    assert 'class="ch-rail"' not in html
    tab_routes = {
        "buscar": "/search",
        "cenas": "/scenes",
        "anotar": "/annotate",
        "rimas": "/rimas",
        "proc": "/processing",
    }
    href = tab_routes[active]
    # The active top tab carries the "tab on" class and targets its route.
    assert 'class="tab on"' in html
    assert href in html


# ── Group 1e: Buscar main pane (Task 10) ──────────────────────────────────────
#
# The Mojica Buscar template rewrites the legacy text/image toggle + raw
# scene-grid into a single ``.b-cp`` section with:
#   * ``.search-wrap`` containing the qbar + 2 modality chips + wired retrieval
#     knobs (Hybrid sem/bm25, k, and Rerank when text),
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
    # 2 modality chips visible. The labels are translated; the
    # ``data-mode`` attribute is stable across locales.
    for mode in ("text", "image"):
        assert f'data-mode="{mode}"' in html
    # Both modality chips are live by default. The disabled affordance
    # still exists in the template — it's gated on the per-modality
    # ``cfg.search.*_enabled`` flags and re-engages if a flag flips to
    # false. (The audio + fusion chips were removed with the audio
    # feature, R2.)
    for mode in ("text", "image"):
        chip = re.search(rf'<button[^>]*data-mode="{mode}"[^>]*>', html)
        assert chip and " disabled" not in chip.group(0), f"chip {mode!r} unexpectedly disabled"
    # Retrieval knob row — only backed controls are visible. Hybrid + k +
    # Rerank are interactive Alpine popovers. Hybrid Search plan Task E2
    # moved the sem/bm25 readout from a server-rendered float to a
    # client-computed ``x-text`` driven by the buscarRetrieval store,
    # so the bare ``sem 0.70`` substring no longer ships from the server.
    assert "knob-popover" in html
    assert 'data-state="readonly"' not in html
    assert "Rerank" in html
    assert "MMR" not in html
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


def test_left_pane_collection_counts_follow_scene_buckets(seed_metadata):
    """Collection counts use the same tipo buckets as the Scenes filters."""
    seeded = seed_metadata()
    from api.services.chrome_service import build_chrome_context

    ctx = build_chrome_context(seeded["cfg"])
    counts = {c["category"]: c["count"] for c in ctx["collections"]}
    assert counts[None] == 2
    assert counts["exterior"] == 2
    assert counts["cartela"] == 0
    assert counts["dialogo"] == 0
    assert counts["interior"] == 0


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
    must produce its empty-state placeholder rather than crashing on an
    undefined ``selected_scene`` reference or leaving a 380px void on the
    right of the canvas.
    """
    r = client.get("/search")
    assert r.status_code == 200, r.text[:500]
    # The inspector partial is included by search.html and now renders a
    # ``.b-rp.b-rp-empty`` placeholder when no scene is selected. Since U6 it
    # also carries the shared ``.fx-empty`` primitive, so assert on the class
    # token rather than the exact attribute string. The full ``.b-rp`` chrome
    # (htabs, .insp-kf, .b-thread, …) must NOT be present.
    assert "b-rp-empty" in r.text, "empty-state placeholder missing"
    assert "fx-empty" in r.text, "empty-state did not adopt the shared .fx-empty primitive"
    assert 'class="htabs"' not in r.text, "selected-state htabs leaked into empty render"
    assert 'class="insp-kf"' not in r.text, "selected-state keyframe leaked into empty render"


# ── Task E2: Buscar toolrow interactive popovers ──────────────────────────────
#
# The Hybrid Search plan replaces the read-only retrieval knob row in
# Buscar with Alpine popovers driving the ``buscarRetrieval`` store +
# hidden HTMX-mirror inputs inside ``#search-text-form``. The two tests
# below pin the markup contract so future template churn cannot silently
# break the popover↔HTMX wiring.


def test_search_toolrow_renders_hybrid_popover(client) -> None:
    """``/tab/search`` exposes the interactive Hybrid + k popovers and
    the hidden HTMX mirrors that ride the existing ``hx-include`` scope."""
    resp = client.get("/tab/search")
    assert resp.status_code == 200
    body = resp.text
    # Popover toggle button with Alpine binding to the store.
    assert "$store.buscarRetrieval" in body
    # Three radio options for retrieval mode.
    assert 'value="clip"' in body
    assert 'value="bm25"' in body
    assert 'value="hybrid"' in body
    # Slider for sem_w (range input).
    assert 'type="range"' in body
    # Changing backed retrieval controls refreshes the active query after
    # Alpine has copied the new values into the hidden HTMX mirrors.
    assert "refreshSearch()" in body
    assert 'x-ref="searchInput"' in body
    # Hidden mirrors INSIDE #search-text-form for HTMX include.
    assert 'name="retriever"' in body
    assert 'name="sem_w"' in body
    assert 'name="bm25_w"' in body
    assert 'name="top_k"' in body


def test_search_toolrow_wires_rerank_and_hides_search_mmr(client) -> None:
    """Buscar exposes backed text rerank but keeps Search-tab MMR hidden."""
    resp = client.get("/tab/search")
    body = resp.text
    assert 'name="reranker_enabled"' in body
    assert "$store.buscarRetrieval.rerank_enabled" in body
    assert "Rerank" in body
    assert '@change="refreshSearch()"' in body
    assert "MMR" not in body
    assert 'data-state="readonly"' not in body


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
        * ``properties``  → now merged into annotate_tags.html, so props dl
                            is visible under tab=annotations too.

    Unknown values must fall back to ``annotations`` via
    :func:`api.services.annotations.normalize_annotate_tab`; assert that
    too so a future regression in the normaliser does not silently break
    the default-tab landing state.
    """
    seed_metadata()

    # comments tab still routes (disabled in UI, but the route is valid).
    r = client.get("/api/annotate/scene?id=351&tab=comments")
    assert r.status_code == 200
    assert 'class="a-rp"' in r.text

    r = client.get("/api/annotate/scene?id=351&tab=annotations")
    assert r.status_code == 200
    # Annotations sub-partial preserves the legacy tag-editor input.
    assert "annotate-tags-input" in r.text
    # AI description + properties dl are now part of the annotations tab.
    assert "moondream-2" in r.text or "No LLM description" in r.text
    assert 'class="props"' in r.text

    r = client.get("/api/annotate/scene?id=351&tab=properties")
    assert r.status_code == 200
    # properties is now an alias for annotations — same content.
    assert "annotate-tags-input" in r.text

    # Unknown tabs fall back to annotations (which shows annotate-tags-input).
    r = client.get("/api/annotate/scene?id=351&tab=bogus")
    assert r.status_code == 200
    assert "annotate-tags-input" in r.text


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
        from api.services.processing_render import render_stepper

        job = inject_job()
        job.progress = 0.4
        html = render_stepper(job)
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
    # Tipo pill carries one of the canonical tipo modifier classes
    # (tinted-chip styling lives in cenas.css `.tipo-pill--<tipo>`, matching
    # the Mojica prototype; replaced the old inline `var(--c-cat-…)` style).
    assert "tipo-pill--" in html
    # And the legacy v0.3 marker is gone from the swap target.
    assert 'class="scene-card"' not in html


def test_tipo_classifier_unit():
    """``tipo_of`` returns the documented bucket for each tag pattern."""
    from api.services.scenes import tipo_of

    assert tipo_of(["cartela", "title-card"], None) == "cartela"
    assert tipo_of(["interior", "baixa-luz"], None) == "interior"
    assert tipo_of(["exterior", "rural"], None) == "exterior"
    assert tipo_of(["duas-pessoas"], None) == "dialogo"
    assert tipo_of([], None) == "transicao"
    # Description-driven cartela fallback (no matching tag).
    assert tipo_of([], "title sequence") == "cartela"


# ── Cenas toolrow: Group by + Sort by params ─────────────────────────────────
#
# The Cenas toolrow's Group / Sort popovers were inert prior to this work —
# decorative spans with no popover, no hidden inputs, no backend params.
# These tests pin (1) the toolrow exposes the hidden ``name="group"`` /
# ``name="sort"`` inputs inside ``#scenes-toolrow`` so the existing
# hx-include scope picks them up on every grid refresh; (2) the popover
# radio rows render; (3) ``/api/scenes?sort=duration`` reorders the cards;
# (4) ``/api/scenes?group=tipo`` emits one ``.group`` heading per tipo;
# (5) ``/api/scenes?group=none`` emits NO ``.group`` headings (flat).


def test_scenes_toolrow_carries_group_and_sort_hidden_inputs(client, seed_metadata):
    """Hidden ``name=group`` and ``name=sort`` ride along on every refresh.

    The ``.find`` form's ``hx-include="#scenes-toolrow"`` is what propagates
    keyword + tags + group + sort on a single keyup. If the hidden inputs
    aren't inside the toolrow, ``?group=…&sort=…`` never reaches the
    service layer and the user's choice is lost on the next refresh.
    """
    seed_metadata()
    r = client.get("/scenes")
    assert r.status_code == 200, r.text[:300]
    html = r.text
    assert 'id="scenes-toolrow"' in html
    # The hidden inputs live INSIDE the toolrow (any string match within
    # the response is sufficient — the integration test below proves the
    # value actually reaches the backend).
    assert 'name="group"' in html
    assert 'name="sort"' in html
    # The two new popovers exist + carry the prototype's radio rows.
    assert 'id="scenes-group-popover"' in html
    assert 'id="scenes-sort-popover"' in html
    assert 'name="group_choice"' in html
    assert 'name="sort_choice"' in html
    # The toolrow owns the refresh request. Radio changes trigger it only
    # after Alpine updates the hidden mirrors, avoiding stale group/sort
    # query params.
    assert 'hx-trigger="refresh"' in html
    assert 'hx-target="#scenes-grid"' in html
    assert "refreshScenes()" in html


def test_api_scenes_sort_duration_reorders_cards(client, seed_metadata):
    """``?sort=duration`` puts the longer scene first (within its group)."""
    seed_metadata()
    # Default sort=timecode → scene 351 (start 83s) before scene 352 (120s).
    r = client.get("/api/scenes?sort=timecode")
    assert r.status_code == 200, r.text[:300]
    html_default = r.text
    idx_351 = html_default.find('data-scene-id="351"')
    idx_352 = html_default.find('data-scene-id="352"')
    assert idx_351 >= 0 and idx_352 >= 0
    assert idx_351 < idx_352, "default sort=timecode should keep 351 ahead of 352"

    # sort=duration → scene 352 (8s) ahead of scene 351 (7s).
    r = client.get("/api/scenes?sort=duration")
    assert r.status_code == 200
    html_dur = r.text
    idx_351 = html_dur.find('data-scene-id="351"')
    idx_352 = html_dur.find('data-scene-id="352"')
    assert idx_351 >= 0 and idx_352 >= 0
    assert idx_352 < idx_351, "sort=duration should put the longer scene first"


def test_api_scenes_sort_pins_falls_back_to_timecode_when_tied(client, seed_metadata):
    """Tied pin_count (both 0) falls back to start_s, preserving timecode order."""
    seed_metadata()
    r = client.get("/api/scenes?sort=pins")
    assert r.status_code == 200
    html = r.text
    idx_351 = html.find('data-scene-id="351"')
    idx_352 = html.find('data-scene-id="352"')
    assert idx_351 < idx_352, "tied pin_count should tie-break by start_s"


def test_api_scenes_group_none_drops_headings(client, seed_metadata):
    """``?group=none`` renders scenecards in a single flat list — no .group bars."""
    seed_metadata()
    # Baseline: default group=film emits exactly one .group heading
    # (one film registered in the seed).
    r = client.get("/api/scenes?group=film")
    assert r.status_code == 200
    html_film = r.text
    assert html_film.count('class="group"') == 1
    # Flatten: no headings, but scenecards still render.
    r = client.get("/api/scenes?group=none")
    assert r.status_code == 200
    html_none = r.text
    assert html_none.count('class="group"') == 0
    assert 'class="scenecard"' in html_none
    # Cards still resolve their inspector URL via per-scene ``film_slug``.
    assert "/api/scenes/351/inspector?film=default" in html_none


def test_api_scenes_group_tipo_emits_tipo_headings(client, seed_metadata):
    """``?group=tipo`` swaps film headings for tipo headings.

    Both seeded scenes carry the ``exterior`` tag → they share the
    "Exterior" tipo heading. The heading dot pulls from the
    ``--c-cat-exterior`` palette variable (not the per-film accent).
    """
    seed_metadata()
    r = client.get("/api/scenes?group=tipo")
    assert r.status_code == 200, r.text[:300]
    html = r.text
    # At least one .group heading lands; its dot carries the tipo
    # CSS variable so the heading colour matches the scenecard pills.
    assert 'class="group"' in html
    assert "var(--c-cat-exterior)" in html
    # The heading label is the tipo's display name, not a film title.
    assert "Exterior" in html
    # And the legacy film title is no longer in the heading row
    # (the test fixture's film is "Default Film" — it can still
    # appear in the per-card sub line via ``s.film.title``, but the
    # ``.group`` heading carries the tipo label).
    group_start = html.find('class="group"')
    next_close = html.find("</div>", group_start)
    heading_html = html[group_start:next_close]
    assert "Default Film" not in heading_html


def test_api_scenes_bucket_filters_by_tipo(client, seed_metadata):
    """Left-pane collection shortcuts use ``?bucket=`` to filter scene tipos."""
    seed_metadata()

    r = client.get("/api/scenes?bucket=exterior")
    assert r.status_code == 200, r.text[:300]
    html = r.text
    assert 'class="scenecard"' in html
    # Exterior scene rendered with its tipo modifier class (tinted-chip
    # styling in cenas.css; replaced the old inline `var(--c-cat-exterior)`).
    assert "tipo-pill--exterior" in html

    r = client.get("/api/scenes?bucket=interior")
    assert r.status_code == 200, r.text[:300]
    assert "No scenes match the filters." in r.text
    assert 'class="scenecard"' not in r.text


def test_api_scenes_unknown_group_falls_back_to_film(client, seed_metadata):
    """A stray ``?group=foobar`` is normalised to the default, not a 500."""
    seed_metadata()
    r = client.get("/api/scenes?group=foobar&sort=lolwhat")
    assert r.status_code == 200
    html = r.text
    # Default group=film semantics restored: one .group heading.
    assert html.count('class="group"') == 1
    assert "Default Film" in html


def test_scene_dict_carries_start_s_and_duration_s(client, seed_metadata):
    """``_card_to_scene`` adds the raw seconds Sort-by-Duration depends on.

    Service-layer unit test (no template render) — pins the contract
    so a future refactor that drops ``start_s`` / ``duration_s`` from
    the card → scene mapping fails loud here instead of silently
    breaking the Sort popover.
    """
    seed_metadata()
    from api.deps import get_config
    from api.services.scenes import build_cenas_context

    ctx = build_cenas_context(get_config())
    scenes = ctx["groups_by_film"][0]["scenes"]
    assert {s["id"] for s in scenes} == {351, 352}
    for s in scenes:
        assert "start_s" in s and isinstance(s["start_s"], float)
        assert "duration_s" in s and isinstance(s["duration_s"], float)
        # Each scene also carries its own film namespace for cross-film
        # groupings (group=tipo / group=none).
        assert s["film"] is not None
        assert s["film"].slug == "default"


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
    assert 'hx-target="closest .tab-panel"' in html
    assert 'hx-swap="outerHTML"' in html


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


def test_proc_active_step_renders_resource_metrics(client, inject_job, monkeypatch):
    """The Processing right pane renders backed resource metrics when active."""
    import api.services.processing_service as processing_service

    monkeypatch.setattr(
        processing_service,
        "build_resource_metrics",
        lambda: [{"label": "CPU", "value": 0.42}],
    )
    inject_job()
    r = client.get("/processing")
    assert r.status_code == 200, r.text[:500]
    html = r.text
    assert "RESOURCES" in html
    assert "CPU" in html
    assert "42%" in html


def test_pipeline_start_response_refreshes_full_processing_tab(client, seed_metadata, monkeypatch):
    """Starting a job must return the full tab so live regions mount."""
    import api.jobs as jobs
    import api.routes.processing as processing

    seeded = seed_metadata()
    raw = seeded["cfg"].paths.library_dir / "default" / "raw" / "default.mp4"

    def fake_start_job(video_path: str, enabled_steps: set[str], cfg) -> str:
        job = jobs.JobState(
            id="started1",
            video_path=video_path,
            status=jobs.STATUS_CREATED,
            steps=[jobs.StepInfo(name=name, label=label) for name, label in jobs.STEP_DEFS],
        )
        job.steps[0].state = "active"
        jobs._registry.add(job)
        return job.id

    monkeypatch.setattr(processing, "start_job", fake_start_job)

    resp = client.post(
        "/api/pipeline/start",
        data={"video_path": str(raw), "steps": ["scene_detection"]},
    )

    assert resp.status_code == 200, resp.text[:500]
    html = resp.text
    assert 'class="p-cp"' in html
    assert 'id="processing-job"' in html
    assert 'id="proc-log"' in html
    assert 'sse-connect="/api/pipeline/stream/started1"' in html
    assert 'hx-swap-oob="innerHTML"' in html


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


# ── Task 26 ───────────────────────────────────────────────────────────
# Phase 7 polish layer: `polish.css` ships the toast / palette / help /
# poke-chip styles up front; `mojica.js` is extended with a `ToastBus`
# IIFE that listens for ``HX-Trigger: {"toast": {...}}`` server events
# and renders `.toast` cards inside the `#toast-root` div from
# ``base.html``. These three tests pin the asset-serving + template-
# wiring contracts in pytest (browser-level animation behaviour stays
# out of scope until Playwright lands in Phase 9).


def test_polish_css_served(client):
    """``/static/css/polish.css`` is served and contains the toast scaffolding.

    The asset is a vanilla stylesheet vendored under ``web/static/css``
    so we can assert on substrings of the source. The `.toast-host`
    selector and `.toast` base class are the structural anchors the
    ToastBus JS depends on; either marker is sufficient evidence the
    file is the Phase-7 polish layer (not, e.g., an empty placeholder).
    """
    r = client.get("/static/css/polish.css")
    assert r.status_code == 200, r.text[:200]
    body = r.text
    assert ".toast" in body
    assert "toast-host" in body
    # Animations the ToastBus JS relies on.
    assert "p-toast-in" in body
    assert "p-toast-out" in body


def test_mojica_js_contains_toast_bus(client):
    """``/static/js/mojica.js`` exposes ``ToastBus`` and targets ``#toast-root``.

    Pins the public contract: ``window.ToastBus`` exists, and the IIFE
    appends `.toast` elements to the `#toast-root` div rendered by
    ``base.html``. Both markers must be present so a future refactor
    that renames either side breaks loudly.
    """
    r = client.get("/static/js/mojica.js")
    assert r.status_code == 200, r.text[:200]
    body = r.text
    assert "ToastBus" in body
    assert "toast-root" in body
    # The HX-Trigger ⇒ "toast" event listener is the wiring the server
    # helper depends on; pin it so the server contract can't drift away
    # from the client one.
    assert "'toast'" in body or '"toast"' in body


@pytest.mark.parametrize(
    "path",
    ["/search", "/scenes", "/annotate", "/rimas", "/processing"],
)
def test_toast_root_present_on_each_tab(client, path):
    """Every Mojica tab full-page render carries the ``#toast-root`` mount.

    The div lives in ``base.html`` so the assertion is structural — any
    full-page route that goes through the chrome shell must ship the
    mount, otherwise ToastBus calls become silent no-ops. The polish
    stylesheet is linked from the same head, so we also smoke-test
    the link element here.
    """
    r = client.get(path)
    assert r.status_code == 200, r.text[:300]
    html = r.text
    assert 'id="toast-root"' in html
    assert "css/polish.css" in html


def test_toast_trigger_helper_sets_hx_trigger_header():
    """``api.deps.toast_trigger`` serialises the spec into ``HX-Trigger``.

    The helper is the canonical server-side entry point: any route that
    wants to surface a toast on the client should call it instead of
    hand-writing the JSON. This unit test pins the wire format the JS
    bus listens for (``{"toast": {...}}``) and the optional-field
    behaviour (``sub`` / ``duration`` are omitted when ``None``).
    """
    import json as _json

    from fastapi import Response

    from api.deps import toast_trigger

    # Minimal spec: only title + default kind.
    r = Response()
    toast_trigger(r, title="Saved")
    payload = _json.loads(r.headers["HX-Trigger"])
    assert "toast" in payload
    assert payload["toast"] == {"title": "Saved", "kind": "info"}

    # Full spec: all optional fields populated.
    r2 = Response()
    toast_trigger(
        r2,
        title="Tags saved",
        sub="2 tags · scene 351",
        kind="success",
        duration=5000,
    )
    payload2 = _json.loads(r2.headers["HX-Trigger"])
    assert payload2["toast"] == {
        "title": "Tags saved",
        "kind": "success",
        "sub": "2 tags · scene 351",
        "duration": 5000,
    }


# ── U7: success/confirmation states (toast bus wiring + inline ✓) ─────────────
#
# The ``toast_trigger`` helper had 0 callers before U7. These tests pin the
# end-to-end wiring: the mutating routes (add-film + the annotate save /
# description-save / tag-curation routes) now set the ``HX-Trigger`` toast
# header on success, AND the annotate responses still carry the inline ✓
# confirmation row. The header is parsed back to the exact wire shape the JS
# ToastBus consumes (``{"toast": {...}}``).


def _toast_payload(response):
    """Parse the ``HX-Trigger`` header into the ``toast`` spec dict.

    Returns ``None`` when the header is absent or carries no ``toast`` key.
    Mirrors the client-side contract: htmx fires a ``toast`` CustomEvent
    whose ``detail`` is the inner object (see ``mojica.js`` ToastBus).
    """
    import json as _json

    raw = response.headers.get("HX-Trigger")
    if not raw:
        return None
    return _json.loads(raw).get("toast")


def test_add_film_success_redirects_to_processing(tmp_config, client):
    """``POST /api/library/add`` redirects to the Processing tab on success.

    After registering the film and enqueuing it, the route returns an
    HX-Redirect to ``/processing?film=<slug>`` so the user lands on the
    Processing tab where they can see and start the pending job.
    """
    from pathlib import Path

    raw_dir = Path(tmp_config.paths.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    video = raw_dir / "novo_filme.mp4"
    video.touch()

    r = client.post(
        "/api/library/add",
        data={"video_path": str(video), "title": "Novo Filme"},
    )
    assert r.status_code == 200, r.text[:300]
    redirect = r.headers.get("HX-Redirect", "")
    assert redirect.startswith(
        "/processing"
    ), f"add-film success must HX-Redirect to /processing, got: {redirect!r}"
    assert "novo_filme" in redirect, "redirect URL should contain the film slug"


def test_add_film_failure_does_not_emit_toast(client):
    """A failed add (missing video) re-renders the form WITHOUT a toast.

    The error path returns the add-film form with a U1 inline field error;
    no success toast must fire (the toast is a success-only affordance).
    """
    r = client.post(
        "/api/library/add",
        data={"video_path": "/does/not/exist.mp4", "title": "Ghost"},
    )
    assert r.status_code == 200, r.text[:300]
    assert _toast_payload(r) is None


def test_annotate_save_emits_toast_and_inline_check(seed_metadata, client):
    """``POST /api/annotate/save`` carries BOTH the toast and the inline ✓.

    U7 requires the inline confirmation (next to the saved control) in
    addition to the global toast. The inline ✓ is the
    ``annotate-feedback--success`` row rendered on ``saved=True``; the
    toast is the ``HX-Trigger`` header.
    """
    seed_metadata()
    r = client.post(
        "/api/annotate/save",
        data={"scene_id": 351, "filter": "all", "tags": "rural"},
    )
    assert r.status_code == 200, r.text[:300]
    # Inline ✓ (the existing success row) — still present.
    assert "annotate-feedback--success" in r.text
    assert "✓" in r.text  # the ✓ glyph
    # Global toast.
    toast = _toast_payload(r)
    assert toast is not None, "annotate save must emit an HX-Trigger toast"
    assert toast["kind"] == "success"
    assert toast["title"] == "Saved"


def test_annotate_description_save_emits_toast_and_inline_check(seed_metadata, client):
    """``POST /api/annotate/description`` carries the toast + the inline ✓ row."""
    seed_metadata()
    r = client.post(
        "/api/annotate/description",
        data={
            "scene_id": 351,
            "filter": "all",
            "description": "a quiet rural road at dawn",
            "tab": "comments",
        },
    )
    assert r.status_code == 200, r.text[:300]
    # Inline "✓ Description saved" row (rendered on desc_saved=True).
    assert "annotate-feedback--success" in r.text
    assert "Description saved" in r.text
    toast = _toast_payload(r)
    assert toast is not None
    assert toast["kind"] == "success"
    assert toast["title"] == "Saved"


def test_annotate_tag_delete_emits_toast(seed_metadata, client):
    """The tag-curation routes (delete) emit a success toast via ``_saved_scene``.

    Scene 352 carries a seeded manual tag ``manual-only``; deleting it
    exercises the shared ``_saved_scene`` helper that fires the toast for
    all three curation routes (delete / rename / AI-tag suppress).
    """
    seed_metadata()
    r = client.post(
        "/api/annotate/tag/delete",
        data={
            "scene_id": 352,
            "tag": "manual-only",
            "filter": "all",
            "tab": "annotations",
        },
    )
    assert r.status_code == 200, r.text[:300]
    toast = _toast_payload(r)
    assert toast is not None
    assert toast["kind"] == "success"
    assert toast["title"] == "Saved"


def test_toast_bus_consumes_hx_trigger_event(client):
    """Prove the ToastBus consumes the exact ``HX-Trigger`` the helper sets.

    The server helper writes ``HX-Trigger: {"toast": {...}}``. htmx turns the
    top-level key into a CustomEvent named ``toast`` whose ``detail`` is the
    inner object. ``mojica.js`` must (a) register a body listener for the
    ``'toast'`` event and (b) forward ``evt.detail`` into ``ToastBus.push``.
    This pins the JS side so the round trip is real end-to-end, not just a
    header that nothing reads.
    """
    body = client.get("/static/js/mojica.js").text
    # The listener is bound on document.body for the 'toast' event.
    assert "addEventListener('toast'" in body
    # …and forwards the event detail straight into the bus.
    assert "ToastBus.push(evt && evt.detail)" in body


# ── Phase 7 / Task 28: keyboard help overlay (?) ─────────────────────────────
#
# The help overlay scaffold lives in `partials/_help_overlay.html` and is
# included from `base.html` (replacing the Phase-1 `#help-root` placeholder).
# These tests pin the structural contract: every full-page tab carries the
# scaffold, the navigation legend documents the five tabs, and `mojica.js`
# exposes the `window.Help` toggle the `?` keypress drives.


@pytest.mark.parametrize(
    "path",
    ["/search", "/scenes", "/annotate", "/rimas", "/processing"],
)
def test_help_overlay_present_on_every_tab(client, path):
    """The help overlay scaffold ships on every full-page render.

    Like the toast mount, the overlay is server-rendered into `base.html`
    so the open path (toggle `[hidden]` on `#help`) costs zero round
    trips. We assert the outer `.kh-back` element exists with the right
    id, and that it is hidden by default (the `hidden` HTML5 attribute
    is what `window.Help.open()` flips).
    """
    r = client.get(path)
    assert r.status_code == 200, r.text[:300]
    html = r.text
    assert 'id="help"' in html
    # The outer node carries the polish-css class + ARIA role.
    assert 'class="kh-back"' in html
    assert 'role="dialog"' in html
    # `hidden` (HTML5 boolean attribute) starts the overlay collapsed.
    # Jinja renders bare booleans as the attribute name only — we accept
    # either `hidden` (correct HTML5) or `hidden=""` (some frameworks).
    assert " hidden" in html or 'hidden="' in html


def test_help_overlay_documents_navigation_shortcuts(client):
    """The help overlay's Navigation column lists every Mojica tab.

    The legend must stay in sync with the chrome's five tabs — otherwise
    a user pressing `?` to discover the `1`..`5` shortcuts would see a
    stale picture. We assert the section heading is present and that
    every PT/EN tab label appears at least once inside the overlay
    markup. Using full-page `/search` keeps the assertion grounded in
    what the browser renders, not a partial fragment.
    """
    r = client.get("/search")
    html = r.text
    # Section heading (rendered inside `<h3>` under `.kh-group`).
    assert "Navigation" in html or "Navegação" in html
    # All five tab labels surface as `desc` rows. We check the EN labels
    # (the test client defaults to `en` for unauthenticated sessions);
    # any of the PT alternates may also appear if the locale flipped.
    for label in ("Search", "Scenes", "Annotate", "Visual rhymes", "Processing"):
        assert label in html or label.replace("Visual rhymes", "Rimas visuais") in html


def test_mojica_js_contains_help_toggle(client):
    """`/static/js/mojica.js` exposes `window.Help` and the `?` handler.

    Pins the public JS contract: the file exports a `Help` namespace
    with `open`/`close`/`toggle` so other surfaces (a future TopBar
    "?" button, palette command "Show shortcuts") can drive the
    overlay without duplicating the toggle logic. The `?` keypress
    handler must be present too — if a future refactor accidentally
    drops it, the keyboard route silently breaks.
    """
    r = client.get("/static/js/mojica.js")
    assert r.status_code == 200, r.text[:200]
    body = r.text
    # The public surface.
    assert "window.Help" in body
    assert "openHelp" in body
    assert "closeHelp" in body
    assert "toggleHelp" in body
    # The keypress wiring. Single-quoted '?' is the canonical form in
    # the IIFE; we also accept double-quoted in case a future formatter
    # rewrites the literal.
    assert "'?'" in body or '"?"' in body


def test_mojica_js_registers_buscar_retrieval_store(client) -> None:
    """The Alpine store name + persist key + defaults are pinned strings.

    Task E1 of the Hybrid Search plan: the Buscar knob row's popovers
    (E2) bind to ``Alpine.store('buscarRetrieval')``, persisted under
    ``mojica:buscar:retrieval``. Defaults match the search route's
    canonical hybrid baseline (``retriever=hybrid`` / ``sem_w=0.70``) so
    a first-paint UI never drifts from the server contract. ``bm25_w`` is
    derived client-side as ``1 - sem_w`` in the hidden HTMX mirror (E3
    follow-up: single source of truth for the weight pair), so the store
    no longer carries it as a persisted field. The ``top_k`` field is the
    UI-preferred value (9) and is intentionally distinct from the route's
    FastAPI default (8) — the hidden HTMX mirror (E2) sends the UI value
    on every request.
    """
    resp = client.get("/static/js/mojica.js")
    assert resp.status_code == 200
    body = resp.text
    # Store name (canonical Alpine pattern).
    assert "Alpine.store('buscarRetrieval'" in body
    # localStorage key.
    assert "mojica:buscar:retrieval" in body
    # Default values that the UI popovers will display.
    assert "'hybrid'" in body  # default retriever mode
    assert "0.70" in body  # default sem_w
    assert "rerank_enabled" in body
