"""U1 — accessible inline form validation + error states.

Covers the three input surfaces (image dropzone, add-film form, search) for:
  * ``aria-invalid="true"`` on the offending field;
  * ``aria-describedby`` pointing at an id that EXISTS in the response;
  * an inline error element with ``role="alert"`` (announced to SR);
  * the ``.is-error`` class on the message;
  * a translated message (en source + pt_BR catalog);
  * the HTMX out-of-band swap mechanism (``hx-swap-oob`` / ``data-field-error``);
  * the search submit-vs-keyup distinction (no error flash mid-typing).

Hermetic: empty temp config, no heavy models (the upload is rejected before
any index load / CLIP forward pass).
"""

from __future__ import annotations

import re
from pathlib import Path

from kuaa.library import register_film

# ── helpers ────────────────────────────────────────────────────────────


def _describedby_target_exists(html: str, field_attr_chunk: str) -> bool:
    """Assert the field's aria-describedby points at an id present in *html*.

    *field_attr_chunk* is a slice of the field tag containing its
    ``aria-describedby="..."``. We extract the referenced id and confirm a
    matching ``id="..."`` exists somewhere in the fragment.
    """
    m = re.search(r'aria-describedby="([^"]+)"', field_attr_chunk)
    assert m, "no aria-describedby on the field"
    target = m.group(1)
    return f'id="{target}"' in html


# ── Image dropzone ─────────────────────────────────────────────────────


def test_image_upload_unsupported_type_returns_accessible_oob_error(client) -> None:
    """A non-image upload → 400 carrying the OOB field-error fragment.

    Status stays 400 (A4 contract); the body is the accessible inline error,
    not the page-level envelope.
    """
    files = {"file": ("notes.txt", b"not an image", "text/plain")}
    r = client.post("/api/search/image", files=files)
    assert r.status_code == 400, r.text
    html = r.text
    assert 'id="image-upload-error"' in html
    assert 'role="alert"' in html
    assert 'class="field-error is-error"' in html
    assert "data-field-error" in html
    assert 'hx-swap-oob="outerHTML"' in html  # lands next to the field, not in #search-results
    # Translated message (en source string):
    assert "Unsupported file type" in html


def test_image_upload_too_large_returns_too_large_message(client) -> None:
    """An oversized upload → 400 with the ``upload_too_large`` message."""
    from api.services import search as search_service

    big = b"\xff" * (search_service.MAX_UPLOAD_BYTES + 1)
    files = {"file": ("huge.jpg", big, "image/jpeg")}
    r = client.post("/api/search/image", files=files)
    assert r.status_code == 400, r.text[:200]
    assert "Image is too large" in r.text
    assert 'id="image-upload-error"' in r.text
    assert 'role="alert"' in r.text


def test_search_html_image_input_has_describedby_to_existing_slot(client) -> None:
    """The Buscar page wires the file input to a real error slot id."""
    r = client.get("/tab/search")
    assert r.status_code == 200
    html = r.text
    # The file input carries aria-describedby + an initial aria-invalid="false".
    m = re.search(r"<input id=\"image-input\".*?>", html, re.DOTALL)
    assert m, "image-input not found"
    chunk = m.group(0)
    assert 'aria-describedby="image-upload-error"' in chunk
    assert 'aria-invalid="false"' in chunk
    assert _describedby_target_exists(html, chunk)


# ── Add-film form ──────────────────────────────────────────────────────


def test_add_film_file_not_found_is_accessible(client) -> None:
    """A path that resolves to no file re-renders the form with an a11y error."""
    r = client.post(
        "/api/library/add",
        data={"video_path": "/definitely/not/here.mp4", "title": ""},
        headers={"HX-Request": "true"},
    )
    assert r.status_code == 200, r.text
    html = r.text
    field = re.search(r"<input[^>]*name=\"video_path\"[^>]*>", html)
    assert field, "video_path field missing"
    chunk = field.group(0)
    assert 'aria-invalid="true"' in chunk
    assert 'aria-describedby="add-film-error"' in chunk
    assert _describedby_target_exists(html, chunk)
    assert 'id="add-film-error"' in html
    assert 'role="alert"' in html
    assert "is-error" in html
    assert "File not found" in html


def test_add_film_duplicate_slug_is_accessible(tmp_config, client) -> None:
    """A duplicate slug → re-rendered form with the already-in-library message.

    The re-add is only blocked when the existing film has a valid symlink.
    Orphan registrations (missing symlink) allow repair; here we create the
    symlink explicitly to simulate a fully-registered, healthy film.
    """
    raw_dir = Path(tmp_config.paths.raw_dir)
    library_dir = Path(tmp_config.paths.library_dir)
    # Create a real file so video.exists() passes, then pre-register its slug.
    (raw_dir / "dup_film.mp4").write_bytes(b"\x00")
    register_film(
        library_dir, slug="dup_film", title="Dup Film", year=None, raw_filename="dup_film.mp4"
    )
    # Also create the symlink so the film is a healthy registration (not an
    # orphan). Without the symlink, re-add would be treated as a repair.
    per_film_raw = library_dir / "dup_film" / "raw"
    per_film_raw.mkdir(parents=True, exist_ok=True)
    (per_film_raw / "dup_film.mp4").symlink_to((raw_dir / "dup_film.mp4").resolve())

    r = client.post(
        "/api/library/add",
        data={"video_path": "dup_film.mp4", "title": ""},
        headers={"HX-Request": "true"},
    )
    assert r.status_code == 200, r.text
    html = r.text
    field = re.search(r"<input[^>]*name=\"video_path\"[^>]*>", html)
    assert field and 'aria-invalid="true"' in field.group(0)
    assert 'id="add-film-error"' in html
    assert 'role="alert"' in html
    assert "is-error" in html
    assert "already in the library" in html


def test_add_film_clean_form_has_no_error_but_keeps_describedby_target(client) -> None:
    """The pristine add-film form: aria-invalid=false, slot present but empty."""
    r = client.get("/api/library/add-form")
    assert r.status_code == 200
    html = r.text
    field = re.search(r"<input[^>]*name=\"video_path\"[^>]*>", html)
    assert field and 'aria-invalid="false"' in field.group(0)
    # The error slot exists (stable describedby target) but carries no is-error.
    assert 'id="add-film-error"' in html
    assert "field-error is-error" not in html


# ── Search query ───────────────────────────────────────────────────────


def test_search_submit_empty_query_surfaces_accessible_error(client) -> None:
    """Submitting an empty query (form trigger) → OOB error into the slot."""
    r = client.get(
        "/api/search",
        params={"q": ""},
        headers={"HX-Request": "true", "HX-Trigger": "search-text-form"},
    )
    assert r.status_code == 200
    html = r.text
    assert 'id="search-query-error"' in html
    assert 'role="alert"' in html
    assert 'class="field-error is-error"' in html
    assert "data-field-error" in html
    assert 'hx-swap-oob="outerHTML"' in html
    assert "Type a query to search" in html


def test_search_submit_one_char_query_says_too_short(client) -> None:
    """A 1-char query on submit → the ``query_too_short`` message."""
    r = client.get(
        "/api/search",
        params={"q": "a"},
        headers={"HX-Request": "true", "HX-Trigger": "search-text-form"},
    )
    assert r.status_code == 200
    assert "at least 2 characters" in r.text
    assert 'class="field-error is-error"' in r.text


def test_search_live_keyup_short_query_is_silent(client) -> None:
    """A live keyup (input trigger) on a short query must NOT flash an error."""
    r = client.get(
        "/api/search",
        params={"q": "a"},
        headers={"HX-Request": "true", "HX-Trigger": "search-input"},
    )
    assert r.status_code == 200
    # The slot is still returned (so a prior error is cleared) but must be empty.
    assert "field-error is-error" not in r.text
    assert 'id="search-query-error"' in r.text


def test_search_no_submit_trigger_is_silent(client) -> None:
    """Any request without the form submit trigger stays silent (no flash).

    A keyup carries ``HX-Trigger: search-input``; a direct/non-HTMX hit sends
    no trigger header at all. Neither equals ``search-text-form``, so neither
    surfaces an error — only an explicit submit does.
    """
    r = client.get(
        "/api/search",
        params={"q": ""},
        headers={"HX-Request": "true", "HX-Trigger": "search-input"},
    )
    assert r.status_code == 200
    assert "field-error is-error" not in r.text
    # No trigger header at all → also silent.
    r2 = client.get("/api/search", params={"q": "a"})
    assert r2.status_code == 200
    assert "field-error is-error" not in r2.text


def test_search_page_input_wires_describedby_to_existing_slot(client) -> None:
    """The Buscar search box wires aria-describedby to a real slot id."""
    r = client.get("/tab/search")
    assert r.status_code == 200
    html = r.text
    field = re.search(r"<input id=\"search-input\".*?>", html, re.DOTALL)
    assert field, "search-input not found"
    chunk = field.group(0)
    assert 'aria-describedby="search-query-error"' in chunk
    assert 'aria-invalid="false"' in chunk
    assert _describedby_target_exists(html, chunk)


# ── i18n: PT catalog resolves for every surface ────────────────────────


def test_field_errors_translate_to_pt(tmp_config) -> None:
    """Each surface's error renders the pt_BR string, not the English source."""
    from fastapi.testclient import TestClient

    from api.server import app

    c = TestClient(app)
    c.__enter__()
    try:
        c.cookies.set("locale", "pt_BR")
        # image upload
        r = c.post("/api/search/image", files={"file": ("x.txt", b"x", "text/plain")})
        assert "Tipo de arquivo não suportado" in r.text
        # add-film not found
        r = c.post(
            "/api/library/add",
            data={"video_path": "/nope/x.mp4"},
            headers={"HX-Request": "true"},
        )
        assert "Arquivo não encontrado" in r.text
        # search empty submit
        r = c.get(
            "/api/search",
            params={"q": ""},
            headers={"HX-Request": "true", "HX-Trigger": "search-text-form"},
        )
        assert "Digite uma busca para pesquisar" in r.text
    finally:
        c.__exit__(None, None, None)
