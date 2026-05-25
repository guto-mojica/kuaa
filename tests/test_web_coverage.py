"""
tests/test_web_coverage.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Phase 2 of the FastAPI regression-recovery effort: web test coverage
expansion.

This module is *characterization / coverage*, not bug reproduction. It
pins the current, real behaviour of the four tabs and the library
sidebar across the empty-data and seeded-data states, plus the
annotation save/clear write path and the corrupt search-index path.

Everything here runs on the consolidated isolation fixtures in
``tests/conftest.py`` (``client``, ``seed_metadata``, ``inject_job``):
no GPU, no CLIP model load, no real video, no network, and — enforced
by ``tmp_config``'s in-fixture path guard — zero reads/writes against
the repository ``data/`` directory.

Groups:

  1. No-data state — Search / Scenes / Annotate / Processing, both the
     full page (``/x``) and the HTMX fragment (``/tab/x``), return 200
     and show the established empty-state marker.
  2. Seeded scenes — the generic ``seed_metadata`` dataset renders
     cards / tags through the scenes route.
  3. Annotation save & clear — exercises the POST routes and asserts the
     on-disk ``manual_annotations.json``. The STR scene-id keys and the
     lower-kebab tag normalization / empty-fragment drop are produced
     ROUTE-SIDE (``api/routes/annotate.py``), not by ``annotator.save``
     (which only ``json.dump``s the dict it is handed).
  4. Library inventory / honest global state — v0.3 is SINGLE-FILM
     (Phase-5 maintainer decision). The sidebar is a plain raw-video
     inventory plus ONE honest global artifact-state summary; there is
     no per-film selection and no fabricated per-file scene counts. The
     old Phase-2 tripwire that pinned the misleading ``{slug}/select``
     placeholder is CONVERTED here to assert the new honest contract
     (the route is gone; the inventory carries no clickable select).
  5. Corrupt index — embeddings row count != mapping row count. Current
     code does NOT validate this; the search path crashes. Captured via
     ``xfail(strict=True)`` so Phase 3c flips it.
  6. Image-upload rejection — a POST to /api/search/image whose body
     ``validate_upload`` rejects must degrade to HTTP 200 + the
     upload-error notice (the ``{% elif upload_error %}`` branch), not
     500. This is the only Phase-3c behaviour without a route-level
     test; it pins the full multipart -> validate_upload ->
     UploadRejected -> upload_error context -> template wiring.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

# Empty-state markers, lifted from the partials so they stay the single
# canonical strings the Phase 0/1a tests also key on.
SEARCH_NO_INDEX = "No search index found. Run the pipeline with the Embeddings step first."
SCENES_NO_DATA = "No scenes found. Run the pipeline with the Scene Detection step first."
ANNOTATE_NO_DATA = "Run the Scene Detection step first."
PROCESSING_NO_JOBS = "No active jobs."
LIBRARY_NO_FILMS = "No films in library"


# ── Group 1: no-data state for every tab ──────────────────────────────────────


class TestNoDataState:
    """Empty temp dirs → every tab responds 200 with its empty marker.

    The full-page parity bug is owned by Phase 1a (already fixed: those
    routes now share the tab context builders), so the full page and the
    fragment must BOTH show the marker. Search is the exception: the
    no-index hint lives in ``search_results.html`` which is only
    rendered by the ``/api/search`` data endpoint, not by the
    tab/full-page shell — so Search's empty state is asserted via
    ``/api/search``.
    """

    def test_scenes_tab_fragment_empty(self, client):
        r = client.get("/tab/scenes")
        assert r.status_code == 200, r.text[:300]
        assert SCENES_NO_DATA in r.text

    def test_scenes_full_page_empty(self, client):
        r = client.get("/scenes")
        assert r.status_code == 200, r.text[:300]
        assert "<!DOCTYPE html>" in r.text
        assert SCENES_NO_DATA in r.text

    def test_annotate_tab_fragment_empty(self, client):
        r = client.get("/tab/annotate")
        assert r.status_code == 200, r.text[:300]
        assert ANNOTATE_NO_DATA in r.text

    def test_annotate_full_page_empty(self, client):
        r = client.get("/annotate")
        assert r.status_code == 200, r.text[:300]
        assert "<!DOCTYPE html>" in r.text
        assert ANNOTATE_NO_DATA in r.text

    def test_processing_tab_fragment_empty(self, client):
        r = client.get("/tab/processing")
        assert r.status_code == 200, r.text[:300]
        assert PROCESSING_NO_JOBS in r.text

    def test_processing_full_page_empty(self, client):
        r = client.get("/processing")
        assert r.status_code == 200, r.text[:300]
        assert "<!DOCTYPE html>" in r.text
        assert PROCESSING_NO_JOBS in r.text

    def test_search_tab_fragment_responds_empty(self, client):
        # The search tab shell itself has no no-index hint (it renders an
        # empty results container); the hint is produced by /api/search.
        r = client.get("/tab/search")
        assert r.status_code == 200, r.text[:300]
        assert 'class="tab-panel"' in r.text

    def test_search_full_page_responds_empty(self, client):
        r = client.get("/search")
        assert r.status_code == 200, r.text[:300]
        assert "<!DOCTYPE html>" in r.text

    def test_search_query_with_no_index_shows_hint(self, client):
        """A real query against empty embeddings dir → no_index branch.

        ``_load_index`` returns ``(None, None, None)`` when the .npy /
        mapping files are absent (empty temp embeddings_dir), and the
        route renders ``search_results.html`` with ``no_index=True``.
        Query must be >= 2 chars or the route short-circuits to "".
        """
        r = client.get("/api/search", params={"q": "horse"})
        assert r.status_code == 200, r.text[:300]
        assert SEARCH_NO_INDEX in r.text

    def test_short_query_returns_empty_body(self, client):
        """< 2 chars short-circuits before any index touch (no crash,
        empty body) — documents the guard, CLIP-free."""
        r = client.get("/api/search", params={"q": "a"})
        assert r.status_code == 200
        assert r.text == ""


# ── Group 2: seeded scenes render ─────────────────────────────────────────────


class TestSeededScenes:
    """The generic seed_metadata dataset flows through the scenes route."""

    def test_scenes_tab_not_empty_state(self, seed_metadata, client):
        seed_metadata()
        r = client.get("/tab/scenes")
        assert r.status_code == 200, r.text[:300]
        assert SCENES_NO_DATA not in r.text
        # 2 seeded scenes → Task 15's countrow renders the total as a
        # ``<span class="v">N</span>`` value followed by the localised
        # "scenes" label. The legacy ``<p>N scenes</p>`` count line is
        # gone with the Mojica rewrite.
        assert '<span class="v">2</span>' in r.text

    def test_scenes_grid_renders_both_scene_ids(self, seed_metadata, client):
        seed_metadata()
        r = client.get("/api/scenes")
        assert r.status_code == 200, r.text[:300]
        # scene 351 has timecode_start "00:01:23"; scene 352 "00:02:00".
        assert "00:01:23" in r.text
        assert "00:02:00" in r.text

    def test_scenes_tag_filter_intersects(self, seed_metadata, client):
        """``dia`` is an LLM tag on scene 351 only (int id) → filtering
        returns just that scene's timecode, not 352's."""
        seed_metadata()
        r = client.get("/api/scenes", params={"tags": ["dia"]})
        assert r.status_code == 200, r.text[:300]
        assert "00:01:23" in r.text
        assert "00:02:00" not in r.text

    def test_scenes_manual_str_tag_matches_int_scene(self, seed_metadata, client):
        """``manual-only`` is a manual annotation on STR key "352"; the
        keyframe scene_id is INT 352. The Phase-1c canonical-key fix
        means this still matches → scene 352 only."""
        seed_metadata()
        r = client.get("/api/scenes", params={"tags": ["manual-only"]})
        assert r.status_code == 200, r.text[:300]
        assert "00:02:00" in r.text
        assert "00:01:23" not in r.text

    def test_scenes_keyword_filter(self, seed_metadata, client):
        """Keyword searches the description blob. 'office' is only in
        scene 352's description."""
        seed_metadata()
        r = client.get("/api/scenes", params={"q": "office"})
        assert r.status_code == 200, r.text[:300]
        assert "00:02:00" in r.text
        assert "00:01:23" not in r.text


# ── Group 3: annotation save / clear write path ───────────────────────────────


class TestAnnotationSaveClear:
    """The POST save/clear routes and the resulting on-disk JSON.

    The on-disk shape (STR scene-id key, lower-kebab tags, empty
    fragments dropped) is produced by the *route*
    (``api/routes/annotate.py``: ``ann[str(scene_id)] = [...]`` with the
    normalizing list-comp); ``cinemateca.annotator.save`` just
    ``json.dump``s. These tests assert that route-side contract.

    Hermetic: the annotations file lives in the temp metadata dir
    (asserted outside the repo by tmp_config's guard). filter is forced
    to ``"all"`` so the saved scene stays in the scene_list and the
    returned partial reflects the new tags (the default ``no_llm``
    filter drops scenes that have a valid LLM description; the seeded
    scenes have descriptions so they'd be filtered out).
    """

    def test_save_writes_normalized_tags_to_disk(self, seed_metadata, client):
        manual_path: Path = seed_metadata()["manual_path"]

        # Input has an empty fragment (",,") and a trailing comma so the
        # route's ``if t.strip()`` drop branch is exercised by data, not
        # just by claim.
        r = client.post(
            "/api/annotate/save",
            data={
                "scene_id": 351,
                "filter": "all",
                "tags": "Rural,, Open Field , exterior,",
            },
        )
        assert r.status_code == 200, r.text[:300]

        on_disk = json.loads(manual_path.read_text())
        # The scene-id key is a STRING (annotator just json.dumps the
        # dict the route built with ``ann[str(scene_id)] = ...``). The
        # lower-kebab normalization and empty-fragment drop happen
        # ROUTE-SIDE (api/routes/annotate.py:
        # ``[t.strip().lower().replace(" ", "-") for t in
        # tags.split(",") if t.strip()]``); ``annotator.save`` does no
        # transformation.
        assert "351" in on_disk
        assert on_disk["351"] == ["rural", "open-field", "exterior"]
        # The empty fragments were dropped (not stored as "").
        assert "" not in on_disk["351"]
        # Pre-existing seeded annotation for scene 352 must survive.
        assert on_disk["352"] == ["manual-only", "noite"]

    def test_save_feedback_in_response(self, seed_metadata, client):
        seed_metadata()
        r = client.post(
            "/api/annotate/save",
            data={"scene_id": 351, "filter": "all", "tags": "rural"},
        )
        assert r.status_code == 200
        assert "annotate-feedback--success" in r.text

    def test_clear_removes_only_target_scene(self, seed_metadata, client):
        manual_path: Path = seed_metadata()["manual_path"]

        # Seed an annotation on 351 first so there is something to clear.
        client.post(
            "/api/annotate/save",
            data={"scene_id": 351, "filter": "all", "tags": "rural"},
        )
        assert "351" in json.loads(manual_path.read_text())

        r = client.post(
            "/api/annotate/clear",
            data={"scene_id": 351, "filter": "all"},
        )
        assert r.status_code == 200, r.text[:300]

        on_disk = json.loads(manual_path.read_text())
        assert "351" not in on_disk
        # The unrelated seeded annotation must remain.
        assert on_disk["352"] == ["manual-only", "noite"]

    def test_clear_missing_scene_is_noop_not_error(self, seed_metadata, client):
        """Clearing a scene with no annotation must not 500 (ann.pop has
        a default) and must leave the file otherwise intact."""
        manual_path: Path = seed_metadata()["manual_path"]
        r = client.post(
            "/api/annotate/clear",
            data={"scene_id": 351, "filter": "all"},
        )
        assert r.status_code == 200
        on_disk = json.loads(manual_path.read_text())
        assert on_disk == {"352": ["manual-only", "noite"]}


class TestAnnotateSceneEmptyFilterRegression:
    """Regression lock: ``/api/annotate/scene`` must not 500 when the
    default ``no_llm`` filter empties the scene list.

    Both seeded scenes have valid LLM descriptions, so the route's
    default ``filter=no_llm`` yields an empty list. ``build_annotate_context``
    (the /tab/annotate path) already falls back to ``filter=all`` in that
    case, but ``build_scene_panel`` (the /api/annotate/scene HTMX-nav path)
    did not — so ``scene_context`` returned the empty-shape dict WITHOUT
    ``current_idx`` and ``annotate_scene.html`` raised
    ``jinja2.UndefinedError: 'current_idx' is undefined`` → HTTP 500.
    Surfaced by a full-library regen (every scene then has a description).
    """

    def test_annotate_scene_default_filter_does_not_500(self, seed_metadata, client):
        seed_metadata()
        r = client.get("/api/annotate/scene?id=351")
        assert r.status_code == 200, r.text[:300]
        # Renders a real scene panel via the all_done→"all" fallback,
        # not the undefined-variable crash.
        assert "annotate-nav__pos" in r.text
        assert "1 / 2" in r.text


# ── Group 4: library filter / select ──────────────────────────────────────────


def _register_film_in_tmp(library_dir: Path, slug: str, title: str) -> Path:
    """Register a film and create its per-film ``raw/`` directory layout.

    Returns the created per-film raw video path.  The video file is
    ``touch``ed (0 bytes) — no route opens it for these sidebar tests.
    """
    from cinemateca.library import register_film

    register_film(
        library_dir,
        slug=slug,
        title=title,
        year=None,
        raw_filename=f"{slug}.mp4",
    )
    raw_dir = library_dir / slug / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_video = raw_dir / f"{slug}.mp4"
    raw_video.touch()
    return raw_video


class TestLibrary:
    """``/api/library/filter`` + the multi-film library sidebar contract.

    T9 rewrote this class to use the registry-backed multi-film layout.
    ``scan_library`` reads ``films.json``; ``library_state`` aggregates
    across all registered films. Films are registered via
    ``_register_film_in_tmp``; per-film artefacts live under
    ``library_dir/<slug>/``. There is NO ``/api/library/{slug}/select``
    route — the per-film affordance is deferred to T10.
    """

    def test_filter_empty_library_shows_no_films(self, client):
        r = client.get("/api/library/filter")
        assert r.status_code == 200, r.text[:300]
        assert LIBRARY_NO_FILMS in r.text

    def test_filter_lists_video_in_raw_dir(self, client):
        cfg = _cfg_from_client()
        _register_film_in_tmp(Path(cfg.paths.library_dir), slug="jeca_tatu", title="Jeca Tatu")
        r = client.get("/api/library/filter")
        assert r.status_code == 200, r.text[:300]
        # The title was explicitly registered as "Jeca Tatu".
        assert "Jeca Tatu" in r.text
        assert LIBRARY_NO_FILMS not in r.text

    def test_filter_query_narrows_results(self, client):
        cfg = _cfg_from_client()
        library_dir = Path(cfg.paths.library_dir)
        _register_film_in_tmp(library_dir, slug="jeca_tatu", title="Jeca Tatu")
        _register_film_in_tmp(library_dir, slug="limite", title="Limite")
        r = client.get("/api/library/filter", params={"q": "limite"})
        assert r.status_code == 200, r.text[:300]
        assert "Limite" in r.text
        assert "Jeca Tatu" not in r.text

    def test_inventory_is_not_clickable_and_carries_no_fake_per_film_state(self, client):
        """T9 update of the CONVERTED Phase-2 tripwire.

        Old contract (v0.3 single-film, now removed): no clickable per-film
        ``/api/library/select/{slug}`` navigates via HX-Redirect; the old
        ``/api/library/<slug>/select`` URL shape is gone (404). Per-film
        scene counts are real (read from keyframes_metadata.json).

        Contract asserted here:
          * old URL shape /api/library/<slug>/select → 404
          * new route /api/library/select/<slug> → 200 + HX-Redirect header
          * tree node carries hx-get pointing to the new route
          * per-film scene count badge is shown (real, not fabricated)
        """
        cfg = _cfg_from_client()
        library_dir = Path(cfg.paths.library_dir)
        _register_film_in_tmp(library_dir, slug="jeca_tatu", title="Jeca Tatu")

        # Write per-film metadata so scene count is real.
        meta_dir = library_dir / "jeca_tatu" / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)
        (meta_dir / "keyframes_metadata.json").write_text(
            json.dumps([{"scene_id": i} for i in (1, 2, 3)])
        )

        # Old URL shape is gone.
        gone = client.get("/api/library/jeca_tatu/select")
        assert gone.status_code == 404

        # New route returns HX-Redirect to the film's scenes tab.
        sel = client.get("/api/library/select/jeca_tatu")
        assert sel.status_code == 200
        assert "HX-Redirect" in sel.headers

        r = client.get("/api/library/filter")
        assert r.status_code == 200, r.text[:300]
        assert "Jeca Tatu" in r.text
        assert 'hx-get="/api/library/select/jeca_tatu"' in r.text
        assert "tree-node--active" not in r.text

    def test_sidebar_reports_honest_global_state(self, client):
        """The sidebar surfaces the AGGREGATE artifact state derived from
        per-film ``keyframes_metadata.json`` files (not fabricated).

        T9 fixture rework: registers a film via the registry and uses the
        per-film directory layout instead of the flat raw_dir/metadata_dir.
        Three states are verified in order:
          1. No films registered → "No source video".
          2. Film registered + raw video on disk, no metadata → "Not yet processed".
          3. Metadata seeded with 4 scenes → "Library processed · 4 scenes".
        """
        # No registered films → "no source video".
        r0 = client.get("/api/library/filter")
        assert r0.status_code == 200
        assert "No source video" in r0.text

        cfg = _cfg_from_client()
        library_dir = Path(cfg.paths.library_dir)
        _register_film_in_tmp(library_dir, slug="jeca_tatu", title="Jeca Tatu")

        # Raw video present but unprocessed → "not yet processed".
        r1 = client.get("/api/library/filter")
        assert "Not yet processed" in r1.text
        assert "Library processed" not in r1.text

        # Write per-film keyframes_metadata.json with 4 scenes → processed.
        meta_dir = library_dir / "jeca_tatu" / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)
        (meta_dir / "keyframes_metadata.json").write_text(
            json.dumps([{"scene_id": i} for i in (1, 2, 3, 4)])
        )
        r2 = client.get("/api/library/filter")
        assert "Library processed" in r2.text
        assert "4" in r2.text  # the true aggregate scene count
        assert "Not yet processed" not in r2.text


def _cfg_from_client():
    """The active temp config (tmp_config rebound deps.get_config)."""
    import api.deps as deps

    return deps.get_config()


# ── Group 5: corrupt search index (Phase 3c tripwire) ─────────────────────────


def _write_corrupt_index(cfg):
    """Embeddings .npy with MORE rows than the mapping declares.

    3 embedding rows vs 2 keyframe/scene-id entries. ``CLIPEmbedder.load``
    performs NO length-consistency check, so this loads "successfully"
    and ``SemanticSearch.by_text`` then does
    ``np.argsort(similarities)`` over 3 elements and indexes ``kf_df``
    (2 rows) at position 2 → pandas ``IndexError: single positional
    indexer is out-of-bounds``. Verified empirically before writing this.
    """
    emb_dir = Path(cfg.paths.embeddings_dir)
    np.save(emb_dir / cfg.embeddings.filename, np.eye(4, dtype="float32")[:3])
    mapping = {
        "model": "stub",
        "dimension": 4,
        "total_vectors": 2,
        "normalized": True,
        "keyframe_paths": ["frames/a.jpg", "frames/b.jpg"],
        "scene_ids": [1, 2],
    }
    (emb_dir / cfg.embeddings.mapping_filename).write_text(json.dumps(mapping))


class _StubEmbedder:
    """CLIP-free stand-in. ``by_text`` calls ``encode_text`` before the
    out-of-bounds ``iloc`` that is the actual crash site, so a real CLIP
    load would otherwise happen on the route path. Stubbing here keeps
    the corrupt-index test hermetic (no model download, no GPU) while
    still exercising the production search route end to end."""

    def encode_text(self, query):
        return np.ones(4, dtype="float32")


@pytest.fixture()
def stub_search_embedder(monkeypatch):
    """Make the search service build a CLIP-free embedder on the OK path.

    ``api.services.search._load_and_validate`` does ``embedder =
    CLIPEmbedder()`` on a well-formed index, and the route then calls
    ``searcher.by_text`` → ``encode_text`` → real ``_load_model()``.
    (The on-disk index is loaded/validated by the mtime/size-keyed
    cache in ``api/services/search.py``; ``tmp_config`` /
    ``clear_index_cache()`` keep that cache from leaking across tests —
    there is no ``search._load_index`` seam any more, it was extracted
    into the service in Phase 3c.) Patch ``CLIPEmbedder`` *in the
    embeddings module* (where the service resolves it via ``from
    cinemateca.embeddings import CLIPEmbedder``) so ``.load`` keeps its
    real (defective, no-validation) behaviour but the constructed
    embedder is CLIP-free."""
    import cinemateca.models.clip.openclip as oc

    real_load = oc.OpenClipEmbedder.load

    class _PatchedEmbedder:
        def __init__(self, *a, **k):
            pass

        load = staticmethod(real_load)

        def encode_text(self, query):
            return np.ones(4, dtype="float32")

    monkeypatch.setattr(oc, "OpenClipEmbedder", _PatchedEmbedder)
    return _PatchedEmbedder


def test_corrupt_index_degrades_gracefully(client, stub_search_embedder):
    """A length-mismatched index must NOT crash the search request.

    Phase 3c (api/services/search.py) added load-time shape validation:
    a mapping with fewer keyframe rows than the embeddings matrix is
    classified ``IndexStatus.CORRUPT`` and the route renders the
    no-index / corrupt empty state (HTTP 200 + the no-index hint)
    instead of 500-ing with a pandas ``IndexError``.

    This was an ``xfail(strict=True)`` Phase-2 tripwire; Phase 3c (the
    owning phase) removed the marker and promoted it to a plain
    assertion per the Phase-0 module-docstring convention. Hermetic:
    the embedder is stubbed (no CLIP load); only ``CLIPEmbedder.load``
    keeps its real no-validation behaviour, so the validation under
    test is genuinely the service's, not the AI core's."""
    cfg = _cfg_from_client()
    _write_corrupt_index(cfg)

    r = client.get("/api/search", params={"q": "horse"})
    assert r.status_code == 200, r.text[:300]
    assert SEARCH_NO_INDEX in r.text


def test_corrupt_index_root_defect_still_in_ai_core_but_caught_by_service(
    client,
):
    """Regression story, made explicit and self-documenting.

    The ROOT defect is unchanged and intentionally so: the AI core
    ``CLIPEmbedder.load`` still performs NO row-count check (Phase 3c
    deliberately did NOT touch ``src/cinemateca/embeddings.py`` to avoid
    changing the model/artefact contract), and ``SemanticSearch.by_text``
    over a mismatched index still raises out of pandas. This pins that
    so a future change to the AI core is noticed.

    The FIX lives one layer up: ``api.services.search.load_index``
    validates shape and returns ``IndexStatus.CORRUPT`` *before* any
    ``SemanticSearch`` is constructed, so the crash never reaches a
    request. This test asserts BOTH halves — the core still crashes
    when used raw, AND the service refuses the same corrupt index
    gracefully — which is the correct post-Phase-3c contract (replacing
    the old always-passing 'current behaviour is a crash' pin).
    """
    import pandas as pd

    from cinemateca.library import FilmContext
    from api.services.search import IndexStatus, load_index
    from cinemateca.embeddings import SemanticSearch
    from cinemateca.models.clip.openclip import OpenClipEmbedder

    cfg = _cfg_from_client()
    _write_corrupt_index(cfg)

    emb_path = Path(cfg.paths.embeddings_dir) / cfg.embeddings.filename
    map_path = Path(cfg.paths.embeddings_dir) / cfg.embeddings.mapping_filename
    embeddings, _mapping, kf_df = OpenClipEmbedder.load(emb_path, map_path)
    # AI core load() still silently accepts the mismatch — unchanged.
    assert embeddings.shape[0] == 3
    assert len(kf_df) == 2

    class _StubEmbedder:
        def encode_text(self, q):
            return np.ones(4, dtype="float32")

    # Raw AI-core search over the mismatched index still crashes ...
    searcher = SemanticSearch(embeddings, kf_df, _StubEmbedder())
    with pytest.raises((IndexError, pd.errors.IndexingError)):
        searcher.by_text("horse", top_k=8)

    # ... but the service layer refuses it gracefully (no crash, typed
    # CORRUPT status) before any SemanticSearch is ever constructed.
    ctx = FilmContext.from_config(cfg)
    index = load_index(
        ctx,
        mapping_filename=cfg.embeddings.mapping_filename,
        embeddings_filename=cfg.embeddings.filename,
    )
    assert index.status is IndexStatus.CORRUPT
    assert not index.ok


# ── Group 6: image-upload rejection (Phase 3c route-level coverage) ────────────

# The exact string from the ``{% elif upload_error %}`` branch of
# web/templates/partials/search_results.html — kept here as the single
# canonical marker, same convention as SEARCH_NO_INDEX above.
SEARCH_UPLOAD_ERROR = "That image could not be used. Upload a single image file under 8 MB."


def _write_wellformed_index(cfg):
    """A coherent 2-row index so ``load_index`` returns ``IndexStatus.OK``.

    ``/api/search/image`` calls ``load_index`` BEFORE ``validate_upload``
    (api/routes/search.py): a MISSING/CORRUPT index would short-circuit
    to the no-index state and the upload-rejection branch under test
    would never be reached. So a well-formed index must exist on disk.
    Shape mirrors ``test_search_service._write_index``: N embedding rows
    == N keyframe/scene-id mapping rows == declared total_vectors.
    """
    emb_dir = Path(cfg.paths.embeddings_dir)
    np.save(emb_dir / cfg.embeddings.filename, np.eye(4, dtype="float32")[:2])
    mapping = {
        "model": "stub",
        "dimension": 4,
        "total_vectors": 2,
        "normalized": True,
        "keyframe_paths": ["frames/a.jpg", "frames/b.jpg"],
        "scene_ids": [1, 2],
    }
    (emb_dir / cfg.embeddings.mapping_filename).write_text(json.dumps(mapping))


def test_image_search_rejected_upload_degrades_gracefully(client, stub_search_embedder):
    """A POST to /api/search/image that ``validate_upload`` rejects must
    return HTTP 200 + the upload-error notice, NOT 500.

    Pins the full Phase-3c wiring end to end: multipart body ->
    ``search_service.validate_upload`` -> ``UploadRejected`` ->
    ``upload_error`` template context -> the
    ``{% elif upload_error %}`` branch of search_results.html. The
    rejection is deterministic: a ``text/plain`` body with a ``.txt``
    suffix fails the content-type/suffix guard.

    Hermetic with no CLIP forward pass: ``validate_upload`` raises
    BEFORE the ``run_in_executor`` offload (api/routes/search.py), so
    ``search_image``/``encode_*`` is never called. ``load_index`` DOES
    run first and constructs ``CLIPEmbedder()`` on the OK path, so a
    well-formed index plus the ``stub_search_embedder`` fixture (real
    ``.load``, CLIP-free constructed embedder) keeps it model-free."""
    cfg = _cfg_from_client()
    _write_wellformed_index(cfg)

    r = client.post(
        "/api/search/image",
        files={"file": ("note.txt", b"not an image", "text/plain")},
    )
    assert r.status_code == 200, r.text[:300]
    assert SEARCH_UPLOAD_ERROR in r.text
    # Not the no-index branch and not a results card — the upload-error
    # branch specifically.
    assert SEARCH_NO_INDEX not in r.text
    assert "scene-card" not in r.text
