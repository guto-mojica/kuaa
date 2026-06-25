"""Playwright e2e harness — fixtures + axe-core helper (WS-5 / T5).

These tests drive a REAL browser (headless Chromium via Playwright) against a
REAL uvicorn instance serving the REAL demo library (Jeca Tatu + Edwin Porter).
They cover the runtime behaviour the U1/U2/U3 unit tests could only assert
structurally:

  * focus trap + restore-to-trigger for the modal overlays (U3);
  * loading-skeleton visibility while a request is in flight (U2);
  * inline form-error appearance with ``aria-invalid`` + ``role="alert"`` (U1).

Plus a per-tab render smoke (200 + key element + no JS console errors) and the
``run_axe`` helper the next task (U5) uses to assert zero serious/critical
accessibility violations.

Design notes
------------
* **Real server, not TestClient.** The point of the e2e gate is to exercise the
  shipped uvicorn + static-asset + JS path end to end, so the harness boots
  ``kuaa serve --no-reload`` as a subprocess on a free port (cwd = repo
  root, so the default-config ``data/library`` resolves to the real archive)
  and polls ``/health`` until ready. ``--no-reload`` is mandatory: the reloader
  forks a supervised child that a single ``terminate()`` would orphan.
* **Read-only against ``data/``.** Every interaction here either renders
  (search / scene / about) or submits an input the route REJECTS before any
  write (the U1 image upload is a text file → 400 before disk touch). Nothing
  mutates the archive — consistent with the repo-wide "never touch data/" rule.
* **Marker.** Every test is ``@pytest.mark.e2e`` (see ``pytest_collection_modifyitems``
  below, which stamps the marker on the whole package so an individual test
  can't be forgotten) so the suite stays out of the smoke / heavy CI runs.
* **Browser fixtures** come from ``pytest-playwright`` (``page`` is
  function-scoped → a fresh context per test, clean cookies/localStorage).
  Headless is the default; ``--headed`` still works for local debugging.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
AXE_JS = Path(__file__).resolve().parent / "vendor" / "axe.min.js"

# How long to wait for the uvicorn subprocess to answer /health before
# declaring the boot failed. The probe earlier observed ~2 s cold; 60 s is a
# generous ceiling that also absorbs a first-run import of the (heavy) AI stack.
_BOOT_TIMEOUT_S = 60.0
_BOOT_POLL_S = 0.25


def _free_port() -> int:
    """Grab an OS-assigned free TCP port, then release it.

    There is an unavoidable (tiny) race between closing the probe socket and
    uvicorn binding it; in practice the window is microseconds and the port is
    in TIME_WAIT-free state, so a collision is vanishingly unlikely on a dev
    box / CI runner. Good enough for a local manual gate.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_for_health(base_url: str, proc: subprocess.Popen[bytes], log_path: Path) -> None:
    """Poll ``base_url/health`` until it answers 200 or the deadline passes.

    Fails fast (and surfaces the captured server log) if the process exits
    early — otherwise a crashed boot would hang the whole timeout.
    """
    deadline = time.monotonic() + _BOOT_TIMEOUT_S
    health = base_url + "/health"
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"kuaa serve exited early (code {proc.returncode}) before "
                f"answering /health.\n--- server output ---\n{_read_log(log_path)}"
            )
        try:
            with urllib.request.urlopen(health, timeout=2) as resp:  # noqa: S310 (localhost)
                if resp.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, OSError) as exc:
            last_err = exc
        time.sleep(_BOOT_POLL_S)
    raise RuntimeError(
        f"kuaa serve did not become healthy within {_BOOT_TIMEOUT_S}s "
        f"(last error: {last_err}).\n--- server output ---\n{_read_log(log_path)}"
    )


def _read_log(log_path: Path) -> str:
    """Best-effort read of the server's log file, for diagnostics."""
    try:
        return log_path.read_text(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover - diagnostics only
        return "<no output captured>"


@pytest.fixture(scope="session")
def live_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """Boot ``kuaa serve --no-reload`` on a free port; yield its base URL.

    Session-scoped so the (slightly slow) cold boot is paid once for the whole
    e2e run. Torn down with SIGTERM → wait, escalating to kill if it ignores
    the term.

    The subprocess's stdout/stderr go to a TEMP FILE, never an in-process
    ``subprocess.PIPE``. This matters: uvicorn logs every request to stdout, and
    an unread PIPE fills its ~64 KB OS buffer after a few dozen requests — at
    which point the server's next ``write()`` BLOCKS and the whole instance
    wedges mid-session (every later ``goto`` then times out). A file sink is
    drained by the OS, so the server never stalls, and the log stays readable
    for boot diagnostics.
    """
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    log_path = tmp_path_factory.mktemp("e2e-server") / "serve.log"
    log_file = log_path.open("wb")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "kuaa",
            "serve",
            "--no-reload",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=str(REPO_ROOT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    try:
        _wait_for_health(base_url, proc, log_path)
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive
            proc.kill()
            proc.wait(timeout=10)
        finally:
            log_file.close()


# ── pytest-playwright integration ─────────────────────────────────────────


@pytest.fixture(scope="session")
def base_url(live_server: str) -> str:
    """Expose the live-server URL under the name pytest-playwright reads.

    Setting ``base_url`` lets ``page.goto("/search")`` resolve relative paths
    against the running instance, so the tests never hard-code a port.
    """
    return live_server


@pytest.fixture()
def browser_context_args(browser_context_args: dict[str, Any], base_url: str) -> dict[str, Any]:
    """Merge ``base_url`` into every browser context so relative goto() works.

    Overrides pytest-playwright's own ``browser_context_args`` fixture (same
    name = override); we keep whatever it provided and add ``base_url`` +
    ``ignore_https_errors`` (harmless on http localhost, future-proof).
    """
    return {**browser_context_args, "base_url": base_url, "ignore_https_errors": True}


# ── axe-core helper ────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def axe_source() -> str:
    """The vendored axe-core source, read once per session.

    Vendored at ``tests/e2e/vendor/axe.min.js`` (axe-core 4.10.2, downloaded
    from the cdnjs dist) so the a11y gate has no network dependency and a
    pinned, reproducible ruleset.
    """
    if not AXE_JS.exists():  # pragma: no cover - vendoring guard
        raise FileNotFoundError(
            f"vendored axe-core missing at {AXE_JS}. Re-download with:\n"
            "  curl -fsSL -o tests/e2e/vendor/axe.min.js "
            "https://cdnjs.cloudflare.com/ajax/libs/axe-core/4.10.2/axe.min.js"
        )
    return AXE_JS.read_text(encoding="utf-8")


def wait_for_alpine(page: Any, *, timeout: int = 5000) -> None:
    """Block until Alpine has booted and registered its global stores.

    The deferred ``alpine.min.js`` calls ``Alpine.start()`` (which fires the
    ``alpine:init`` listeners in mojica.js that register the palette / help /
    toast stores) at or before ``DOMContentLoaded`` — but a test that navigates
    with ``wait_until="domcontentloaded"`` and immediately drives a keyboard
    shortcut could still race that boot on a slow run. Gating on the
    ``palette`` store's existence makes every Alpine-dependent interaction
    deterministic without an arbitrary sleep.
    """
    page.wait_for_function(
        "() => !!(window.Alpine && window.Alpine.store && window.Alpine.store('palette'))",
        timeout=timeout,
    )


def run_axe(page: Any, axe_source: str, *, context: Any = None) -> list[dict[str, Any]]:
    """Inject axe-core into ``page`` and return its violations list.

    Each violation is the raw axe result object (``id``, ``impact``,
    ``description``, ``help``, ``helpUrl``, ``nodes`` …). U5 filters these to
    serious/critical and asserts the list is empty after its fixes; this
    harness only provides the mechanism.

    ``context`` is an optional axe context selector (e.g. a CSS string or
    include/exclude object); ``None`` audits the whole document.
    """
    page.add_script_tag(content=axe_source)
    # Run axe in the page. ``resultTypes: ['violations']`` keeps the payload
    # small (we don't ship passes/incomplete back across the bridge).
    return page.evaluate(
        """async (ctx) => {
            const opts = { resultTypes: ['violations'] };
            const target = ctx || document;
            const results = await window.axe.run(target, opts);
            return results.violations;
        }""",
        context,
    )


# ── marker stamping ─────────────────────────────────────────────────────────


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Stamp ``@pytest.mark.e2e`` on every test collected under ``tests/e2e/``.

    Belt-and-suspenders: each test file also declares ``pytestmark =
    pytest.mark.e2e`` explicitly (so the marker is visible at the file head),
    but this hook guarantees a newly-added e2e test can never accidentally
    escape the marker and leak into the smoke / ``not e2e`` CI runs.
    """
    e2e_root = Path(__file__).resolve().parent
    for item in items:
        try:
            item_path = Path(str(item.fspath)).resolve()
        except Exception:  # pragma: no cover - path edge cases
            continue
        if e2e_root in item_path.parents or item_path.parent == e2e_root:
            item.add_marker(pytest.mark.e2e)
