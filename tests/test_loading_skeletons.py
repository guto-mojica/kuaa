"""U2 — loading skeletons for the three async panes.

Skeletons are visual, so pytest cannot see them render; it CAN assert the
markup + CSS that make them work:

  * the ``.skeleton`` shimmer primitive lives in ``fx.css`` and is built on
    the U6 motion/colour tokens;
  * a ``prefers-reduced-motion: reduce`` block neutralises the animation
    (the calm static placeholder the U5 a11y gate checks for);
  * the indicator-gating rules wire the skeleton to htmx's request class;
  * each of the three panes (Buscar results · Cenas grid · scene inspector)
    renders a skeleton placeholder element AND at least one trigger that
    points ``hx-indicator`` at it.

Hermetic: empty temp config + the historical ``seed_metadata`` dataset, no
heavy models (every assertion is on rendered HTML / static CSS text).
"""

from __future__ import annotations

import re
from pathlib import Path

CSS = Path("web/static/css/fx.css").read_text(encoding="utf-8")


def _trigger_chunk_for(html: str, indicator_id: str) -> str | None:
    """Return the opening tag (attribute run) of an element whose
    ``hx-indicator`` points at *indicator_id*, or ``None`` if absent.

    Matches from a ``<`` up to the ``>`` that closes the start tag,
    requiring an ``hx-indicator="#<id>"`` somewhere inside it.
    """
    pat = re.compile(
        r"<[^>]*?hx-indicator=\"#" + re.escape(indicator_id) + r"\"[^>]*?>",
        re.DOTALL,
    )
    m = pat.search(html)
    return m.group(0) if m else None


# ── CSS primitive ───────────────────────────────────────────────────────


def test_skeleton_rule_exists_in_fx_css() -> None:
    # The base block + its three layout helpers.
    assert ".skeleton {" in CSS
    assert ".skeleton-grid {" in CSS
    assert ".skeleton-inspector {" in CSS


def test_skeleton_built_on_u6_tokens() -> None:
    """Shimmer reuses the canonical raised/hover colour tokens (no new
    magic colours) — the U6 hygiene contract."""
    # Slice the .skeleton base rule and assert it references the tokens.
    base = CSS.split(".skeleton {", 1)[1].split("}", 1)[0]
    assert "var(--c-raised)" in base
    assert "var(--c-hover)" in base


def test_skeleton_animation_keyframes_exist() -> None:
    assert "@keyframes fx-skeleton-sweep" in CSS
    # The base rule actually drives the sweep animation.
    base = CSS.split(".skeleton {", 1)[1].split("}", 1)[0]
    assert "animation:" in base
    assert "fx-skeleton-sweep" in base


def test_prefers_reduced_motion_block_neutralises_skeleton() -> None:
    """Under reduced motion the sweep is removed and the moving gradient is
    replaced by a flat tint — a static, non-animated placeholder.

    This feeds the U5 a11y gate, so it is asserted structurally here.
    """
    assert "@media (prefers-reduced-motion: reduce)" in CSS
    # Isolate the reduced-motion block and confirm it kills the animation.
    block = CSS.split("@media (prefers-reduced-motion: reduce) {", 1)[1]
    block = block.split("@media", 1)[0]  # stop at the next @media, if any
    assert ".skeleton" in block
    assert "animation: none" in block
    # The sheen gradient is dropped so nothing moves / shimmers.
    assert "background-image: none" in block


def test_indicator_gating_rules_present() -> None:
    """The skeleton containers are hidden by default and revealed only when
    they carry htmx's request class. htmx 2.x adds ``.htmx-request`` to the
    ``hx-indicator`` target itself, so the rule is a self-combinator."""
    assert ".skeleton-grid.htmx-indicator" in CSS
    assert ".skeleton-inspector.htmx-indicator" in CSS
    # Hidden by default …
    assert re.search(r"\.skeleton-grid\.htmx-indicator[^{]*\{\s*display:\s*none", CSS)
    # … shown while the matching request is in flight.
    assert ".skeleton-grid.htmx-indicator.htmx-request" in CSS
    assert ".skeleton-inspector.htmx-indicator.htmx-request" in CSS


# ── Pane 1: Buscar search results ─────────────────────────────────────────


def test_search_results_pane_has_skeleton_wired_to_indicator(client) -> None:
    r = client.get("/tab/search")
    assert r.status_code == 200
    html = r.text
    # The skeleton grid placeholder exists, is decorative, and is the
    # htmx indicator for search.
    assert 'id="search-skeleton"' in html
    assert "skeleton-grid htmx-indicator" in html
    skel = re.search(r'<div id="search-skeleton"[^>]*>', html).group(0)
    assert 'aria-hidden="true"' in skel
    # Card-shaped placeholders so the swap to real .b-card results doesn't jump.
    assert "skeleton-card" in html
    assert "skeleton--kf" in html
    # At least one /api/search trigger points its indicator at the skeleton.
    chunk = _trigger_chunk_for(html, "search-skeleton")
    assert chunk is not None, "no search trigger wired to #search-skeleton"
    assert "/api/search" in chunk or 'id="search-text-form"' in chunk


# ── Pane 2: Cenas scene grid ──────────────────────────────────────────────


def test_scene_grid_pane_has_skeleton_wired_to_indicator(seed_metadata, client) -> None:
    seed_metadata()
    r = client.get("/tab/scenes")
    assert r.status_code == 200
    html = r.text
    assert 'id="scenes-skeleton"' in html
    assert "skeleton-grid htmx-indicator" in html
    skel = re.search(r'<div id="scenes-skeleton"[^>]*>', html).group(0)
    assert 'aria-hidden="true"' in skel
    assert "skeleton-card" in html
    # The grid-refresh triggers (toolrow + .find form) point at the skeleton.
    chunk = _trigger_chunk_for(html, "scenes-skeleton")
    assert chunk is not None, "no scenes trigger wired to #scenes-skeleton"
    assert "/api/scenes" in chunk


# ── Pane 3: scene inspector ───────────────────────────────────────────────


def test_inspector_skeleton_present_on_search_tab(client) -> None:
    r = client.get("/tab/search")
    assert r.status_code == 200
    html = r.text
    assert 'id="inspector-skeleton"' in html
    assert "skeleton-inspector htmx-indicator" in html
    skel = re.search(r'<div id="inspector-skeleton"[^>]*>', html).group(0)
    assert 'aria-hidden="true"' in skel
    # Inspector-shaped: a wide keyframe + property lines (not card grid).
    assert "skeleton--kf-wide" in html
    assert "sk-props" in html
    # The .rp-slot wrapper is the positioning context for the overlay.
    assert "rp-slot" in html


def test_inspector_skeleton_present_on_scenes_tab(seed_metadata, client) -> None:
    seed_metadata()
    r = client.get("/tab/scenes")
    assert r.status_code == 200
    html = r.text
    assert 'id="inspector-skeleton"' in html
    assert "skeleton-inspector htmx-indicator" in html
    assert "rp-slot" in html


def test_search_result_cards_point_at_inspector_skeleton() -> None:
    """Each search-result card opens the inspector and shows its skeleton
    while the inspector fragment loads.

    Renders ``search_results.html`` directly with one synthetic hit (the
    search route's empty-query / no-index paths don't emit cards under the
    hermetic fixtures, and those validation paths are U1's concern, not
    U2's — here we assert the card markup itself carries the wiring)."""
    from api.templates import templates

    html = templates.env.get_template("partials/search_results.html").render(
        results=[
            {
                "scene_id": 351,
                "film_slug": "default",
                "img_url": None,
                "timecode": "00:01:23:00",
                "similarity": 0.42,
                "description": "a river at dusk",
                "tags": ["exterior"],
                "pin_count": 0,
            }
        ],
        films_by_id={},
        selected_scene_id=None,
        highlighted_tags=set(),
    )
    assert "b-card" in html, "expected a result card to render"
    chunk = _trigger_chunk_for(html, "inspector-skeleton")
    assert chunk is not None, "result card not wired to #inspector-skeleton"
    assert "/inspector" in chunk and 'hx-target="#right-pane"' in chunk


def test_scenecard_fragment_points_at_inspector_skeleton(seed_metadata, client) -> None:
    """The /api/scenes grid fragment's scenecards open the inspector with
    the skeleton as their indicator."""
    seed_metadata()
    r = client.get("/api/scenes")
    assert r.status_code == 200
    html = r.text
    assert "scenecard" in html, "expected seeded library to render scenecards"
    chunk = _trigger_chunk_for(html, "inspector-skeleton")
    assert chunk is not None, "scenecard not wired to #inspector-skeleton"
    assert "/inspector" in chunk and 'hx-target="#right-pane"' in chunk
