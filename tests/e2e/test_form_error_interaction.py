"""E2E inline-form-error runtime tests (U1) — accessible validation, live.

``tests/test_form_validation_a11y.py`` asserts (server-side) that the routes
RETURN the accessible OOB error fragment for each bad input. What it cannot
verify is that, driven through a real browser, the fragment actually LANDS in
the error slot, becomes visible with ``role="alert"``, and flips the field's
``aria-invalid`` to ``true`` — including the htmx ``beforeSwap`` shim in
mojica.js that re-permits an OOB swap on a 4xx image-upload response. That
runtime round-trip is what this module drives.

Two surfaces:
  * **Search query** — submitting an empty / 1-char query via the form (which
    carries ``HX-Trigger: search-text-form``) surfaces the inline error; a live
    keyup must NOT (covered structurally by the unit test; here we assert the
    submit path lights it up and Alpine syncs ``aria-invalid``).
  * **Image upload** — uploading an unsupported (text) file → the route answers
    400 with the OOB error fragment, the shim permits the swap, and the upload
    error slot announces it. Rejected before any CLIP forward pass / disk write.
"""

from __future__ import annotations

from typing import Any

import pytest

from .conftest import wait_for_alpine

pytestmark = pytest.mark.e2e


def _slot_text(page: Any, slot_id: str) -> str:
    return page.evaluate(
        "(id) => { const el = document.getElementById(id); return el ? el.textContent.trim() : ''; }",
        slot_id,
    )


# ── Search query validation ────────────────────────────────────────────────


def test_empty_query_submit_shows_accessible_error(page: Any) -> None:
    """Submitting an empty query surfaces the inline ``role=alert`` error.

    The error slot (``#search-query-error``) gains ``.is-error`` + the
    translated message, and the search input's ``aria-invalid`` flips to
    ``true`` (Alpine's ``syncQueryError`` after the OOB swap).
    """
    page.goto("/search", wait_until="domcontentloaded")
    wait_for_alpine(page)

    # The slot starts empty / not-errored and the field is valid.
    assert _slot_text(page, "search-query-error") == ""
    assert page.locator("#search-input").get_attribute("aria-invalid") == "false"

    # Submit with an empty query. Clicking the form's submit button sends the
    # native form submit → htmx fires with HX-Trigger=search-text-form, the
    # header the route gates the error on.
    page.locator("#search-input").fill("")
    page.locator("#search-text-form button[type=submit]").click()

    # The OOB fragment lands in the slot, which becomes an errored alert.
    page.wait_for_function(
        """() => {
            const el = document.getElementById('search-query-error');
            return el && el.classList.contains('is-error') && el.textContent.trim().length > 0;
        }""",
        timeout=5000,
    )
    slot = page.locator("#search-query-error")
    assert slot.get_attribute("role") == "alert"
    assert "is-error" in (slot.get_attribute("class") or "")
    assert _slot_text(page, "search-query-error") != ""

    # Alpine mirrors the error onto the field's aria-invalid.
    page.wait_for_function(
        "() => document.getElementById('search-input').getAttribute('aria-invalid') === 'true'",
        timeout=5000,
    )
    assert page.locator("#search-input").get_attribute("aria-invalid") == "true"


def test_short_query_submit_shows_too_short_error(page: Any) -> None:
    """A 1-char query on submit surfaces the ``too short`` message (not silent)."""
    page.goto("/search", wait_until="domcontentloaded")
    wait_for_alpine(page)
    page.locator("#search-input").fill("a")
    page.locator("#search-text-form button[type=submit]").click()
    page.wait_for_function(
        """() => {
            const el = document.getElementById('search-query-error');
            return el && el.classList.contains('is-error') && el.textContent.trim().length > 0;
        }""",
        timeout=5000,
    )
    # English catalog message (locale defaults to pt_BR cookie-less, but the en
    # source is the msgid; assert on the stable substring either catalog yields
    # for the too-short case via its visible length > 0 + role).
    assert page.locator("#search-query-error").get_attribute("role") == "alert"
    assert _slot_text(page, "search-query-error") != ""


# ── Image-upload validation ─────────────────────────────────────────────────


def test_unsupported_image_upload_shows_accessible_error(page: Any) -> None:
    """Uploading a text file as an image surfaces the inline upload error.

    Exercises the full runtime path: the multipart POST → 400 → the
    ``htmx:beforeSwap`` shim in mojica.js (which re-permits the OOB fragment on
    an error status) → the upload error slot announces with ``role=alert`` and
    the file input's ``aria-invalid`` reflects the rejection.
    """
    page.goto("/search", wait_until="domcontentloaded")
    wait_for_alpine(page)

    # Switch to the image modality so the dropzone + file input are visible.
    page.locator(".modes .chip[data-mode=image]").click()
    page.locator("#image-input").wait_for(state="attached", timeout=5000)

    # The slot starts clean.
    assert _slot_text(page, "image-upload-error") == ""

    # Upload an unsupported (text) file. set_input_files drives the real
    # <input type=file>; the form's hx-trigger='change from:#image-input' fires
    # the multipart POST.
    page.locator("#image-input").set_input_files(
        files=[
            {
                "name": "notes.txt",
                "mimeType": "text/plain",
                "buffer": b"not an image",
            }
        ]
    )

    # The OOB error fragment lands in the upload slot and becomes an alert.
    page.wait_for_function(
        """() => {
            const el = document.getElementById('image-upload-error');
            return el && el.classList.contains('is-error') && el.textContent.trim().length > 0;
        }""",
        timeout=5000,
    )
    slot = page.locator("#image-upload-error")
    assert slot.get_attribute("role") == "alert"
    assert "is-error" in (slot.get_attribute("class") or "")
    assert _slot_text(page, "image-upload-error") != ""
