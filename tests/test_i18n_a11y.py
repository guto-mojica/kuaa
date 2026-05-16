"""Phase 6 — i18n, accessibility, offline-polish regression tests.

Hermetic (empty temp config; no heavy models). Covers, per the Phase-6
acceptance contract:

  * locale switch preserves the page the user was on AND is
    open-redirect safe (a hostile ``Referer`` falls back to ``/``);
  * ``<html lang>`` reflects the active locale;
  * core nav + About + locale links carry real ``href``s so the app
    works with JavaScript disabled (progressive enhancement);
  * pipeline step labels are translated (proving the label is resolved
    from the step *id* via gettext, NOT printed from the unchanged
    ``STEP_DEFS`` Python tuple — Phase-4 contract intact);
  * ZERO external icon-CDN reference remains anywhere in the shipped
    templates/static (fully offline).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from api.deps import locale_to_lang

REPO = Path(__file__).resolve().parent.parent
TEMPLATES = REPO / "web" / "templates"
STATIC = REPO / "web" / "static"

# Hosts that would mean an icon (or anything) is fetched off-box. The
# project must run fully offline, so NONE of these may appear in any
# shipped template or static asset.
_FORBIDDEN_CDN_HOSTS = (
    "unpkg.com",
    "cdn.jsdelivr.net",
    "jsdelivr",
    "cdnjs.cloudflare.com",
    "cdnjs",
    "fonts.googleapis.com",
    "fonts.gstatic.com",
    "phosphoricons.com",
    "@phosphor-icons/web",
)


# ── locale_to_lang unit contract ──────────────────────────────────────


@pytest.mark.parametrize(
    "locale,expected",
    [("pt_BR", "pt-BR"), ("en", "en"), ("xx_YY", "pt-BR"), ("", "pt-BR")],
)
def test_locale_to_lang_mapping(locale, expected):
    assert locale_to_lang(locale) == expected


# ── <html lang> reflects locale ───────────────────────────────────────


def test_html_lang_default_is_pt_br(raw_client):
    """No cookie → default locale pt_BR → lang="pt-BR"."""
    r = raw_client.get("/search")
    assert r.status_code == 200
    assert '<html lang="pt-BR">' in r.text


def test_html_lang_follows_en_cookie(raw_client):
    raw_client.cookies.set("locale", "en")
    r = raw_client.get("/search")
    assert r.status_code == 200
    assert '<html lang="en">' in r.text


def test_html_lang_about_page(raw_client):
    raw_client.cookies.set("locale", "en")
    r = raw_client.get("/about")
    assert r.status_code == 200
    assert '<html lang="en">' in r.text


# ── Progressive enhancement: real hrefs (works JS-off) ────────────────


@pytest.mark.parametrize(
    "href",
    [
        "/search",
        "/scenes",
        "/annotate",
        "/processing",
        "/about",
        "/api/locale/pt_BR",
        "/api/locale/en",
    ],
)
def test_nav_and_about_and_locale_have_real_href(raw_client, href):
    """Every primary action is a real <a href> the browser can follow
    without htmx/JS. We assert the literal href attribute is present in
    the rendered base shell."""
    r = raw_client.get("/search")
    assert r.status_code == 200
    assert f'href="{href}"' in r.text


def test_about_full_page_route_works_js_off(raw_client):
    """The /about href must resolve to a standalone page (the modal-only
    affordance had no JS-off fallback before Phase 6)."""
    raw_client.cookies.set("locale", "en")
    r = raw_client.get("/about")
    assert r.status_code == 200
    assert "<!DOCTYPE html>" in r.text
    assert "Model attributions" in r.text  # en source string
    assert 'href="/"' in r.text  # a real way back, no JS needed


def test_about_full_page_localized_pt(raw_client):
    """Same page, pt_BR cookie → translated credits heading."""
    raw_client.cookies.set("locale", "pt_BR")
    r = raw_client.get("/about")
    assert r.status_code == 200
    assert "Atribuições de modelos" in r.text
    assert '<html lang="pt-BR">' in r.text


def test_about_modal_and_page_share_credits(raw_client):
    """Modal (/api/about) and full page (/about) emit the same credits.

    Both {% include %} the shared partials/_about_credits.html, so the
    model attributions must appear identically on both surfaces.
    """
    raw_client.cookies.set("locale", "en")
    modal = raw_client.get("/api/about").text
    page = raw_client.get("/about").text
    for needle in ("Moondream 2", "CLIP", "YOLOv8", "MTCNN",
                   "Model attributions"):
        assert needle in modal, needle
        assert needle in page, needle


# ── Locale switch preserves path + open-redirect safety ───────────────


def test_locale_switch_preserves_path_htmx(raw_client):
    """HTMX request → 200 + HX-Redirect back to the SAME page (not /)."""
    r = raw_client.get(
        "/api/locale/en",
        headers={"hx-request": "true", "referer": "http://testserver/scenes"},
    )
    assert r.status_code == 200
    assert r.headers["HX-Redirect"] == "/scenes"
    assert "locale=en" in r.headers.get("set-cookie", "")


def test_locale_switch_preserves_path_js_off(raw_client):
    """Plain navigation (no hx-request) → 303 redirect back to the page."""
    r = raw_client.get(
        "/api/locale/pt_BR",
        headers={"referer": "http://testserver/annotate"},
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/annotate"
    assert "locale=pt_BR" in r.headers.get("set-cookie", "")


@pytest.mark.parametrize(
    "evil_referer",
    [
        "http://evil.test/scenes",  # cross-origin host
        "https://evil.test/",  # cross-origin host
        "//evil.test/path",  # protocol-relative
        "https:///etc/passwd",  # scheme, no host
        "http://testserver/\\evil",  # backslash injection
        "http://testserver/scenes\r\nSet-Cookie: x=1",  # CRLF injection
        "http://testserver/unknown-page",  # not an in-app page
        "javascript:alert(1)",  # js scheme
        "ftp://testserver/scenes",  # non-http scheme + foreign-ish
        "",  # empty
    ],
)
def test_locale_switch_open_redirect_is_rejected(raw_client, evil_referer):
    """A hostile/unknown Referer must NEVER become the redirect target;
    it falls back to '/'. Asserted on BOTH transports."""
    r_hx = raw_client.get(
        "/api/locale/en",
        headers={"hx-request": "true", "referer": evil_referer},
    )
    assert r_hx.status_code == 200
    assert r_hx.headers["HX-Redirect"] == "/", evil_referer

    r_plain = raw_client.get("/api/locale/en", headers={"referer": evil_referer})
    assert r_plain.status_code == 303
    assert r_plain.headers["location"] == "/", evil_referer


def test_locale_switch_no_referer_falls_back_to_root(raw_client):
    r = raw_client.get("/api/locale/en", headers={"hx-request": "true"})
    assert r.status_code == 200
    assert r.headers["HX-Redirect"] == "/"


def test_locale_switch_only_known_pages_pass(raw_client):
    """Same-origin but the path must still be a known app page."""
    for path in ("/", "/search", "/scenes", "/annotate", "/processing", "/about"):
        r = raw_client.get(
            "/api/locale/en",
            headers={"hx-request": "true", "referer": f"http://testserver{path}"},
        )
        assert r.headers["HX-Redirect"] == path, path


# ── Step labels are translated (Phase-4 STEP_DEFS untouched) ──────────


def test_step_defs_unchanged():
    """The Python step contract must be byte-identical to the Phase-4
    shape: ids in STEP_ORDER order, opaque (non-translated) labels.
    Translating these in Python instead of the template would silently
    break the SSE/runner tests that build StepInfo from this tuple."""
    from api.jobs import STEP_DEFS

    assert STEP_DEFS == [
        ("frame_extraction", "Frames"),
        ("scene_detection", "Cenas"),
        ("visual_analysis", "Visual"),
        ("embeddings", "Embeddings"),
        ("llm_description", "Descrições"),
    ]


def test_step_labels_render_translated_pt(raw_client):
    """pt_BR processing page shows the *translated* step labels resolved
    from the step id — NOT the opaque STEP_DEFS Python labels."""
    raw_client.cookies.set("locale", "pt_BR")
    r = raw_client.get("/processing")
    assert r.status_code == 200
    # Translated (from _step_labels.html → gettext), id-driven:
    assert "Extração de quadros" in r.text
    assert "Detecção de cenas" in r.text
    assert "Análise visual" in r.text
    # The opaque internal STEP_DEFS labels must NOT leak to the UI:
    assert "Frames<" not in r.text
    assert ">Frames" not in r.text


def test_step_labels_render_translated_en(raw_client):
    raw_client.cookies.set("locale", "en")
    r = raw_client.get("/processing")
    assert r.status_code == 200
    assert "Frame extraction" in r.text
    assert "Scene detection" in r.text
    assert "Visual analysis" in r.text


# ── Fully offline: zero external icon/CDN reference ───────────────────


def _shipped_text_files():
    for base in (TEMPLATES, STATIC):
        for p in base.rglob("*"):
            if p.is_file() and p.suffix in {
                ".html",
                ".css",
                ".js",
                ".svg",
                ".cfg",
            }:
                # Vendored htmx is local; skip the vendored LICENSE text
                # (it's prose, not a network reference).
                yield p


def test_no_external_cdn_reference_anywhere():
    """Guard: NO shipped template/static asset may reference an external
    CDN host. This is the offline-operation contract — the Phosphor web
    component CDN <script> was the specific regression Phase 6 removed."""
    offenders: list[str] = []
    for p in _shipped_text_files():
        if p.name == "LICENSE":
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        for host in _FORBIDDEN_CDN_HOSTS:
            if host in text:
                offenders.append(f"{p.relative_to(REPO)} :: {host}")
    assert not offenders, "External CDN refs found:\n" + "\n".join(offenders)


def test_base_html_has_no_phosphor_cdn_script(raw_client):
    """End-to-end: the rendered base shell must not pull the Phosphor
    web component (or any unpkg asset) over the network."""
    r = raw_client.get("/search")
    assert r.status_code == 200
    assert "unpkg.com" not in r.text
    assert "@phosphor-icons/web" not in r.text
    assert "<ph-icon" not in r.text  # web component fully replaced
    # ...and the inline SVG replacement is actually present:
    assert 'class="ph-icon"' in r.text
    assert "<svg" in r.text


def test_fonts_are_self_hosted_offline():
    """The Google-Fonts @import was the other network dependency the
    offline contract forbids. Fonts must now be vendored locally with a
    local @font-face stylesheet and ZERO gstatic/googleapis refs."""
    css = (STATIC / "css" / "main.css").read_text(encoding="utf-8")
    assert "googleapis.com" not in css
    assert "gstatic" not in css
    fonts_css = STATIC / "fonts" / "ibmplex.css"
    assert fonts_css.is_file()
    fc = fonts_css.read_text(encoding="utf-8")
    assert "gstatic" not in fc and "googleapis" not in fc
    assert "@font-face" in fc
    assert len(list((STATIC / "fonts").glob("*.woff2"))) >= 2


def test_vendored_phosphor_assets_present():
    """The icons were vendored locally (with upstream MIT LICENSE for
    provenance), not left as a dangling reference."""
    icon_dir = STATIC / "icons" / "phosphor"
    assert icon_dir.is_dir()
    assert (icon_dir / "LICENSE").is_file()
    svgs = list(icon_dir.glob("*.svg"))
    # 16 distinct glyphs are used across the templates.
    assert len(svgs) >= 16, [p.name for p in svgs]
    for svg in svgs:
        body = svg.read_text(encoding="utf-8")
        assert "currentColor" in body  # color parity with old icon font
