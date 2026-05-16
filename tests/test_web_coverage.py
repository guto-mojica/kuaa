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
  4. Library filter / select — asserts the ACTUAL current behaviour
     (select is a known Phase-5 placeholder; the test documents that
     so Phase 5 has a tripwire, it does not pre-implement it).
  5. Corrupt index — embeddings row count != mapping row count. Current
     code does NOT validate this; the search path crashes. Captured via
     ``xfail(strict=True)`` so Phase 3c flips it.
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
        # 2 seeded scenes → the grid count line.
        assert "2 scenes" in r.text

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


# ── Group 4: library filter / select ──────────────────────────────────────────

class TestLibrary:
    """``/api/library/filter`` and ``/api/library/{slug}/select``.

    These assert CURRENT behaviour. ``scan_library`` lists video files
    in raw_dir; the empty temp raw_dir → no films. ``{slug}/select`` is
    a known Phase-5 placeholder: it ignores the slug entirely and just
    re-renders the search partial. The tests document exactly that so
    Phase 5 (per-film data model) has a behavioural tripwire — they do
    NOT pre-implement Phase 5.
    """

    def test_filter_empty_library_shows_no_films(self, client):
        r = client.get("/api/library/filter")
        assert r.status_code == 200, r.text[:300]
        assert LIBRARY_NO_FILMS in r.text

    def test_filter_lists_video_in_raw_dir(self, client):
        cfg = _cfg_from_client()
        (Path(cfg.paths.raw_dir) / "jeca_tatu.mp4").touch()
        r = client.get("/api/library/filter")
        assert r.status_code == 200, r.text[:300]
        # scan_library titleizes the stem: "jeca_tatu" -> "Jeca Tatu".
        assert "Jeca Tatu" in r.text
        assert LIBRARY_NO_FILMS not in r.text

    def test_filter_query_narrows_results(self, client):
        cfg = _cfg_from_client()
        for stem in ("jeca_tatu", "limite"):
            (Path(cfg.paths.raw_dir) / f"{stem}.mp4").touch()
        r = client.get("/api/library/filter", params={"q": "limite"})
        assert r.status_code == 200, r.text[:300]
        assert "Limite" in r.text
        assert "Jeca Tatu" not in r.text

    def test_select_is_phase5_placeholder_renders_search_partial(self, client):
        """CURRENT behaviour: select ignores the slug and returns the
        Search tab partial verbatim — it does NOT load that film's data
        (no per-film model yet). Phase 5 owns making this real; if Phase
        5 changes the response this test fails and must be updated then.
        """
        r = client.get("/api/library/does-not-exist/select")
        assert r.status_code == 200, r.text[:300]
        # search.html partial markers; no full HTML doc, no film name.
        assert 'class="tab-panel"' in r.text
        assert "search-mode-toggle" in r.text
        assert "<!DOCTYPE html>" not in r.text
        assert "does-not-exist" not in r.text


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
    """Make the search route's ``_load_index`` build a stub embedder.

    ``api.routes.search._load_index`` does ``embedder = CLIPEmbedder()``
    and the route then calls ``searcher.by_text`` → ``encode_text`` →
    real ``_load_model()``. Patch ``CLIPEmbedder`` *in the embeddings
    module* (where _load_index resolves it via ``from
    cinemateca.embeddings import CLIPEmbedder``) so ``.load`` keeps its
    real (defective, no-validation) behaviour but the constructed
    embedder is CLIP-free."""
    import cinemateca.embeddings as emb

    real_load = emb.CLIPEmbedder.load

    class _PatchedEmbedder:
        def __init__(self, *a, **k):
            pass

        load = staticmethod(real_load)

        def encode_text(self, query):
            return np.ones(4, dtype="float32")

    monkeypatch.setattr(emb, "CLIPEmbedder", _PatchedEmbedder)
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

    from api.services.film_context import FilmContext
    from api.services.search import IndexStatus, load_index
    from cinemateca.embeddings import CLIPEmbedder, SemanticSearch

    cfg = _cfg_from_client()
    _write_corrupt_index(cfg)

    emb_path = Path(cfg.paths.embeddings_dir) / cfg.embeddings.filename
    map_path = Path(cfg.paths.embeddings_dir) / cfg.embeddings.mapping_filename
    embeddings, _mapping, kf_df = CLIPEmbedder.load(emb_path, map_path)
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
