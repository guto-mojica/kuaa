"""The demo path, walked end to end in Portuguese against the real library.

Search -> inspector -> pane tabs -> selection -> Scenes (``sel``) -> Scenes
(film/scene) -> Rhymes -> Annotate. Each step is recorded by name and the test
fails at the end with every failed step listed, so one broken step does not
hide the ones after it.

Strictly read-only. The harness serves the maintainer's real ``data/``, so a
listener fails the test on any request that is not GET/HEAD, apart from the
read-only search queries named in ``_READ_ONLY_POSTS``. Nothing here clicks
Save, Delete, Export EDL or a pre-processing control, and the Annotate tag form
is never touched.

Observations that are not assertions (English leftovers, description position
at smaller viewports, timeline header, click-to-description timing) go to
``demo_report.json`` under ``KUAA_DEMO_OUT`` (default: the pytest tmp dir) and
to stdout.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import pytest

from .conftest import REPO_ROOT

# Collected by the model-free gates too, which have no Playwright installed;
# skip at import instead of failing collection.
expect = pytest.importorskip("playwright.sync_api").expect

pytestmark = pytest.mark.e2e

_QUERIES = ("mar", "rua", "homem")
_TABS = {
    "buscar": "a.tab[data-tab='buscar']",
    "cenas": "a.tab[data-tab='cenas']",
    "rimas": "a.tab[data-tab='rimas']",
    "anotar": "a.tab[data-tab='anotar']",
}
# Paths a POST may reach. These are the search queries the page issues; they
# only read the index. Every other non-GET/HEAD request fails the test.
_READ_ONLY_POSTS = ("/api/search",)
_LOADED = {"width": 1440, "height": 900}
_ENGLISH = re.compile(
    r"\b(Search|Scenes|Annotate|Rhymes|Processing|Library|Load more|Next|Prev|"
    r"Per page|Scope|Clear|Select|Save|Cancel|Back|All films|Showing|selected)\b",
    re.IGNORECASE,
)
_BG_URL = re.compile(r"url\(\s*[\"']?(.*?)[\"']?\s*\)")


# The search form and its input use hx-sync="replace", so a type-ahead request
# is cancelled on purpose when Enter submits the form. The browser reports the
# cancelled GET as ERR_ABORTED and htmx logs it as sendAbort / afterRequest.
# Only exactly that is ignored, and its volume is capped per query submitted.
_ABORT_FAILURE = "net::ERR_ABORTED"
_SEARCH_PATH = "/api/search"
_MAX_ABORTS_PER_QUERY = 3


def _is_search_abort(req: str) -> bool:
    m = re.fullmatch(r"(\S+) \((.*)\)", req)
    return bool(m) and urlparse(m[1]).path == _SEARCH_PATH and m[2] == _ABORT_FAILURE


def _unexplained_logs(logs: list[str], aborted: int) -> list[str]:
    """Console errors not accounted for by ``aborted`` cancelled searches.

    htmx logs the bare event name with no detail, so the request cannot be read
    from the text. Each cancelled request yields one ``htmx:sendAbort`` and one
    ``htmx:afterRequest``; any more than that, or any other text, is kept.
    """
    budget = {"htmx:sendAbort": aborted, "htmx:afterRequest": aborted}
    kept = []
    for text in logs:
        if budget.get(text, 0) > 0:
            budget[text] -= 1
        else:
            kept.append(text)
    return kept


def _slugs() -> list[str]:
    lib = REPO_ROOT / "data" / "library"
    if not lib.is_dir():
        return []
    return sorted(p.name for p in lib.iterdir() if (p / "embeddings").is_dir())


def _msgstr(msgid: str) -> str:
    """Portuguese translation of ``msgid`` from the catalogue."""
    po = REPO_ROOT / "web" / "locales" / "pt_BR" / "LC_MESSAGES" / "messages.po"
    m = re.search(
        r'^msgid "' + re.escape(msgid) + r'"\nmsgstr "(.*)"$', po.read_text("utf-8"), re.M
    )
    assert m, f"{msgid!r} not in pt_BR catalogue"
    return m.group(1)


def _bg_name(style: str | None) -> str:
    m = _BG_URL.search(style or "")
    return unquote(urlparse(m.group(1)).path.rsplit("/", 1)[-1]) if m else ""


def _bg_url(style: str | None) -> str:
    m = _BG_URL.search(style or "")
    return m.group(1) if m else ""


def _bg(loc: Any) -> str:
    return loc.evaluate("e => getComputedStyle(e).backgroundImage")


class _Walk:
    """Browser state, collectors and helpers shared by the steps."""

    def __init__(self, page: Any, base_url: str, out: Path) -> None:
        self.page = page
        self.base_url = base_url
        self.out = out
        self.console_errors: list[str] = []
        self.bad_responses: list[str] = []
        self.failed_requests: list[str] = []
        self.write_requests: list[str] = []
        self.steps: dict[str, str] = {}
        self.report: dict[str, Any] = {"english": {}}
        self.query = ""
        self.searches = 0
        self.slug = ""
        self.scene = ""
        self.kf = ""
        self.scenes_kf = ""
        self.sel: list[tuple[str, str]] = []
        page.on("console", self._console)
        page.on("pageerror", lambda e: self.console_errors.append(f"pageerror: {e}"))
        page.on(
            "response",
            lambda r: (
                self.bad_responses.append(f"{r.status} {r.request.resource_type} {r.url}")
                if r.status >= 400
                else None
            ),
        )
        page.on(
            "requestfailed",
            lambda r: self.failed_requests.append(f"{r.url} ({r.failure})"),
        )
        page.on("request", self._guard)

    def _console(self, m: Any) -> None:
        if m.type == "error":
            self.console_errors.append(m.text)

    def _guard(self, req: Any) -> None:
        if req.method in ("GET", "HEAD"):
            return
        path = urlparse(req.url).path
        if req.method == "POST" and path in _READ_ONLY_POSTS:
            return
        self.write_requests.append(f"{req.method} {req.url}")

    # navigation / search ------------------------------------------------
    def search(self, q: str) -> int:
        self.searches += 1
        self.page.goto("/search")
        self.page.fill("#search-input", q)
        self.page.press("#search-input", "Enter")
        try:
            self.page.wait_for_selector(".b-card", timeout=15000)
        except Exception:
            return 0
        return self.page.locator(".b-card").count()

    def card(self, slug: str, scene: str) -> Any:
        return self.page.locator(f".b-card[data-film-slug='{slug}'][data-scene-id='{scene}']")

    def click_card(self, card: Any) -> float:
        """Click a card; return seconds until its description is visible."""
        t0 = time.monotonic()
        card.click()
        self.page.locator("#right-pane .b-com.ai").first.wait_for(state="visible", timeout=8000)
        return time.monotonic() - t0

    def url_ok(self, url: str) -> int:
        return int(
            self.page.request.get(self.base_url + url if url.startswith("/") else url).status
        )

    def images_ok(self, selector: str) -> list[str]:
        """Problems with the background images of every ``selector`` match."""
        bad: list[str] = []
        els = self.page.locator(selector)
        for i in range(els.count()):
            url = _bg_url(_bg(els.nth(i)))
            if not url:
                bad.append(f"{selector}[{i}] has no background image")
            elif (st := self.url_ok(url)) != 200:
                bad.append(f"{st} {url}")
        return bad

    def pane_kf(self) -> str:
        return _bg_name(_bg(self.page.locator("#right-pane .insp-kf").first))

    def shot(self, name: str) -> str:
        path = self.out / f"{name}.png"
        self.page.screenshot(path=str(path))
        return str(path)

    # observation --------------------------------------------------------
    def scan_english(self, label: str) -> None:
        found = self.page.evaluate(
            """() => {
              const vis = (e) => !!(e.offsetWidth || e.offsetHeight || e.getClientRects().length);
              const out = {text: [], titles: []};
              document.querySelectorAll('button,[role=tab],a,h1,h2,h3,h4,h5,h6').forEach((e) => {
                if (!vis(e)) return;
                const t = (e.innerText || '').trim().replace(/\\s+/g, ' ');
                if (t) out.text.push(t);
              });
              document.querySelectorAll('[title]').forEach((e) => {
                if (vis(e)) out.titles.push(e.getAttribute('title'));
              });
              return out;
            }"""
        )
        self.report["english"][label] = {
            "text": sorted({t for t in found["text"] if _ENGLISH.search(t)}),
            "titles": sorted({t for t in found["titles"] if t and _ENGLISH.search(t)}),
            "all_titles": sorted({t for t in found["titles"] if t}),
        }

    def step(self, name: str, fn: Callable[[], None]) -> None:
        try:
            fn()
            self.steps[name] = "pass"
        except (AssertionError, Exception) as exc:  # noqa: BLE001 - recorded, re-raised later
            self.steps[name] = f"FAIL: {str(exc).strip()[:1500]}"
            try:
                self.report.setdefault("failure_shots", {})[name] = self.shot(
                    "fail_" + name.split(":")[0].replace(" ", "_")
                )
            except Exception:  # noqa: BLE001
                pass


def _walk_steps(w: _Walk) -> None:
    page = w.page
    slugs = _slugs()
    first_film = slugs[0]

    # 1 ---------------------------------------------------------------
    def step1() -> None:
        n = 0
        for q in _QUERIES:
            n = w.search(q)
            if n:
                w.query = q
                break
        if not n:
            pytest.skip("no query returned a card")
        assert n >= 4, f"query {w.query!r} returned {n} cards"
        cards = page.locator(".b-card")
        films = {cards.nth(i).get_attribute("data-film-slug") for i in range(n)}
        assert len(films) >= 2, f"cards come from {films}"
        bad = w.images_ok(".b-card .kf")
        assert not bad, f"keyframes not answered 200: {bad}"
        w.scan_english("search_results")

    w.step("1 search", step1)
    if w.steps["1 search"] != "pass" or not w.query:
        return

    # 2 ---------------------------------------------------------------
    def pick() -> Any:
        cards = page.locator(".b-card")
        for i in range(cards.count()):
            c = cards.nth(i)
            if c.get_attribute("data-film-slug") not in ("", first_film):
                return c
        pytest.skip("no card from a non-first film")

    def step2() -> None:
        c = pick()
        w.slug = c.get_attribute("data-film-slug") or ""
        w.scene = c.get_attribute("data-scene-id") or ""
        w.kf = _bg_name(_bg(c.locator(".kf")))
        assert w.kf, "clicked card has no keyframe name"
        w.click_card(c)
        pane = page.locator("#right-pane")
        expect(page.locator("#right-pane .insp-kf").first).to_be_visible(timeout=3000)
        assert w.pane_kf() == w.kf, f"pane shows {w.pane_kf()!r}, card {w.kf!r}"
        assert pane.locator(".b-com.ai").first.inner_text().strip(), "empty description"
        expect(page.locator("#inspector-skeleton")).not_to_be_visible()
        w.scan_english("search_with_inspector")

        tl = page.locator(".b-tl")
        w.report["timeline"] = {
            "present": tl.count() > 0,
            "header": tl.locator(".ttitle").inner_text() if tl.count() else None,
            "screenshot": w.shot("timeline_after_card_click") if tl.count() else None,
        }

        times = []
        cards = page.locator(".b-card")
        for i in range(min(5, cards.count())):
            times.append(w.click_card(cards.nth(i)))
        w.report["click_to_description_s"] = {
            "samples": [round(t, 3) for t in times],
            "median": round(statistics.median(times), 3),
            "max": round(max(times), 3),
        }
        w.click_card(w.card(w.slug, w.scene))

    w.step("2 card -> pane", step2)

    def sizes() -> None:
        rows = {}
        for name, size in (
            ("1440x900", _LOADED),
            ("1280x720", {"width": 1280, "height": 720}),
            ("1024x700", {"width": 1024, "height": 700}),
        ):
            page.set_viewport_size(size)
            w.click_card(w.card(w.slug, w.scene))
            page.wait_for_timeout(300)
            desc = page.locator("#right-pane .b-com.ai").first
            box, pbox = desc.bounding_box(), page.locator("#right-pane").bounding_box()
            visible = bool(
                box
                and pbox
                and box["y"] >= max(pbox["y"], 0)
                and box["y"] + box["height"] <= min(pbox["y"] + pbox["height"], size["height"])
            )
            rows[name] = {
                "description_fully_visible": visible,
                "description_box": box,
                "pane_box": pbox,
                "screenshot": w.shot(f"pane_{name}"),
            }
        page.set_viewport_size(_LOADED)
        w.report["description_by_viewport"] = rows

    w.step("2b viewports", sizes)

    # 3 ---------------------------------------------------------------
    def step3() -> None:
        w.click_card(w.card(w.slug, w.scene))
        for idx in (0, 1, 2, 0):
            tab = page.locator("#right-pane [role=tab]").nth(idx)
            with page.expect_response(lambda r: "/inspector" in r.url):
                tab.click()
            expect(page.locator("#inspector-skeleton")).not_to_be_visible()
            expect(page.locator("#right-pane [role=tab]").nth(idx)).to_have_attribute(
                "aria-selected", "true"
            )
            assert w.pane_kf() == w.kf, f"tab {idx}: pane shows {w.pane_kf()!r}, want {w.kf!r}"

    w.step("3 pane tabs", step3)

    # 4 ---------------------------------------------------------------
    def step4() -> None:
        page.goto("/search")
        w.search(w.query)
        w.click_card(w.card(w.slug, w.scene))
        before = w.pane_kf()
        inspector_calls: list[str] = []
        page.on(
            "request", lambda r: inspector_calls.append(r.url) if "/inspector" in r.url else None
        )
        cards = page.locator(".b-card")
        w.sel = []
        for i in range(3):
            c = cards.nth(i)
            w.sel.append(
                (c.get_attribute("data-film-slug") or "", c.get_attribute("data-scene-id") or "")
            )
            c.locator(".check").click()
        page.wait_for_timeout(500)
        assert inspector_calls == [], f"tick triggered {inspector_calls}"
        assert w.pane_kf() == before, "right pane changed on tick"
        bar = page.locator(".selection-bar")
        expect(bar).to_be_visible()
        assert bar.locator(".sel-count span").inner_text() == "3"
        texts = [t.strip() for t in bar.locator(".sel-btn").all_inner_texts()]
        want = [_msgstr(k) for k in ("Open in Scenes", "Annotate", "Export EDL", "Clear")]
        assert texts == want, f"buttons {texts} != {want}"
        label = bar.locator(".sel-count").inner_text()
        assert _msgstr("scene(s) selected") in label, f"count label {label!r}"
        assert "scene(s) selected" not in label

    w.step("4 selection bar", step4)

    # 5 ---------------------------------------------------------------
    def step5() -> None:
        page.locator(".selection-bar .sel-btn--primary").click()
        page.wait_for_function("location.pathname === '/scenes'")
        q = parse_qs(urlparse(page.url).query)
        assert "sel" in q, f"no sel in {page.url}"
        page.wait_for_selector(".scenecard", timeout=15000)
        cards = page.locator(".scenecard")
        assert cards.count() == 3, f"{cards.count()} scene cards, want 3"
        got = set()
        for i in range(3):
            c = cards.nth(i)
            film = parse_qs(urlparse(c.get_attribute("hx-get") or "").query).get("film", [""])[0]
            got.add((film, c.get_attribute("data-scene-id")))
        assert got == set(w.sel), f"cards {sorted(got)} != ticked {sorted(w.sel)}"
        bad = w.images_ok(".scenecard .kf")
        assert not bad, f"thumbnails: {bad}"
        w.scan_english("scenes_sel")
        page.evaluate("sessionStorage.clear()")

    w.step("5 open in scenes", step5)

    # 6 ---------------------------------------------------------------
    def to_tab_from_card(slug: str, scene: str, tab: str) -> None:
        w.search(w.query)
        w.click_card(w.card(slug, scene))
        page.click(_TABS[tab])

    def step6() -> None:
        page.evaluate("sessionStorage.clear()")
        to_tab_from_card(w.slug, w.scene, "cenas")
        page.wait_for_url("**/scenes*")
        q = parse_qs(urlparse(page.url).query)
        assert q.get("film") == [w.slug] and q.get("scene") == [w.scene], page.url
        page.wait_for_selector(".scenecard", timeout=15000)
        first = page.locator(".scenecard").first
        first_slug = parse_qs(urlparse(first.get_attribute("hx-get") or "").query)["film"][0]
        group = page.locator(".group .name")
        names = group.all_inner_texts() if group.count() else []
        assert first_slug != first_film, (
            f"first card is from the alphabetically first film {first_film}; groups {names[:3]}"
        )
        mine = page.locator(f".scenecard[data-scene-id='{w.scene}'][hx-get*='film={w.slug}&']")
        assert mine.count() == 1, f"{mine.count()} cards for {w.slug}/{w.scene}"
        w.scenes_kf = _bg_name(_bg(mine.locator(".kf")))
        expect(page.locator("#right-pane h2").first).to_contain_text(
            _msgstr("Scene %(n)s").replace("%(n)s", f"{int(w.scene):03d}"), timeout=5000
        )
        preview = page.locator("#right-pane .preview").first
        url = _bg_url(_bg(preview))
        assert url and w.url_ok(url) == 200, f"inspector keyframe {url!r}"
        w.scan_english("scenes")

    w.step("6 search -> scenes", step6)

    # 7 ---------------------------------------------------------------
    def rhymes_for(slug: str, scene: str) -> bool:
        page.wait_for_url("**/rimas*")
        assert parse_qs(urlparse(page.url).query).get("anchor") == [f"{slug}/{scene}"], page.url
        anchor = page.locator("#rimas-anchor")
        expect(anchor).to_be_visible(timeout=15000)
        assert anchor.get_attribute("data-film-slug") == slug
        assert anchor.get_attribute("data-scene-id") == scene
        try:
            page.wait_for_selector(".r-echo", timeout=15000)
        except Exception:
            return False
        return True

    def step7() -> None:
        page.click(_TABS["rimas"])
        if not rhymes_for(w.slug, w.scene):
            w.report["rhymes_empty_for"] = f"{w.slug}/{w.scene}"
            pytest.skip("rhymes empty state for the primary scene (see report)")
        assert page.locator(".r-echo").count() >= 1
        bad = w.images_ok("#rimas-anchor .kf") + w.images_ok(".r-echo .kf")
        assert not bad, f"rhymes images: {bad}"
        anchor_kf = _bg_name(_bg(page.locator("#rimas-anchor .kf")))
        assert anchor_kf == w.scenes_kf, f"rhymes anchor {anchor_kf!r} vs scenes {w.scenes_kf!r}"
        w.scan_english("rhymes")

    w.step("7 scenes -> rhymes", step7)

    # 8 ---------------------------------------------------------------
    def step8() -> None:
        page.click(_TABS["anotar"])
        page.wait_for_url("**/annotate*")
        q = parse_qs(urlparse(page.url).query)
        assert q.get("film") == [w.slug] and q.get("id") == [w.scene], page.url
        panel = page.locator(".a-rp")
        expect(panel).to_contain_text(f"{int(w.scene):03d}", timeout=10000)
        tag = page.locator("input.annotate-tags-input")
        expect(tag).to_be_visible()
        expect(tag).to_be_enabled()
        w.scan_english("annotate")

    w.step("8 rhymes -> annotate", step8)


@pytest.fixture()
def demo_out(tmp_path: Path) -> Path:
    out = Path(os.environ.get("KUAA_DEMO_OUT") or tmp_path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def test_demo_path(page: Any, base_url: str, demo_out: Path) -> None:
    if len(_slugs()) < 2:
        pytest.skip("library has fewer than two processed films")
    page.context.add_cookies([{"name": "locale", "value": "pt_BR", "url": base_url}])
    page.set_viewport_size(_LOADED)
    w = _Walk(page, base_url, demo_out)
    _walk_steps(w)

    w.report["steps"] = w.steps
    w.report["collected"] = {
        "console_errors": w.console_errors,
        "bad_responses": w.bad_responses,
        "failed_requests": w.failed_requests,
        "write_requests": w.write_requests,
    }
    (demo_out / "demo_report.json").write_text(json.dumps(w.report, indent=2, default=str))
    print("\nDEMO REPORT\n" + json.dumps(w.report, indent=2, default=str))  # noqa: T201

    all_console = list(w.console_errors)
    aborts = [r for r in w.failed_requests if _is_search_abort(r)]
    w.failed_requests = [r for r in w.failed_requests if not _is_search_abort(r)]
    w.console_errors = _unexplained_logs(w.console_errors, len(aborts))
    from collections import Counter

    print("DISTINCT CONSOLE ERRORS", dict(Counter(all_console)))  # noqa: T201

    failed = {k: v for k, v in w.steps.items() if v != "pass"}
    assert not w.write_requests, f"non-read requests sent: {w.write_requests}"
    assert not failed, f"failed steps: {json.dumps(failed, indent=1)}"
    assert not w.console_errors, f"console errors: {w.console_errors}"
    assert not w.bad_responses, f"responses >= 400: {w.bad_responses}"
    assert not w.failed_requests, f"failed requests: {w.failed_requests}"
    assert len(aborts) <= _MAX_ABORTS_PER_QUERY * w.searches, (
        f"{len(aborts)} aborted searches for {w.searches} queries"
    )
