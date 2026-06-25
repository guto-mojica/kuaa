"""Hermetic characterization snapshots for /api/search.

These tests fix the HTML structure /api/search produces under the
``tmp_config`` fixture (no real CLIP model, no real library) BEFORE any
P1 code moves. Every subsequent commit on this branch must keep them
passing — that is the contract that the refactor is behavior-preserving.

Regenerate with ``UPDATE_SNAPSHOTS=1 uv run pytest tests/test_p1_search_snapshot.py``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.server import app
from tests._snapshot import assert_snapshot


def _slim(html: str) -> str:
    """Reduce HTML to a structural fingerprint stable across whitespace edits."""
    return re.sub(r"\s+", " ", html).strip()


@pytest.fixture
def small_library(tmp_config, monkeypatch):
    """A two-film hermetic library with stub CLIP + BM25 corpus.

    Mirrors tests/test_multi_film_search.py's fixture style: real on-disk
    JSON + .npy stubs + monkeypatched embedder.

    The CLIP embedder is stubbed via two surfaces:

      * ``api.services.search._get_embedder`` — used by ``aggregate_search``
        (cross-film path) to encode the query text once.
      * ``kuaa.models.clip.openclip.OpenClipEmbedder`` — replaced with
        a fake that keeps the real ``.load`` staticmethod (the on-disk
        mapping is well-formed) but stubs the constructor + ``encode_*``
        instance methods so ``_load_and_validate`` builds a SearchIndex
        whose embedder produces 4-dim vectors matching the stub .npy.
    """
    cfg = tmp_config
    import sys

    import kuaa.search.aggregate as _csa_ref  # noqa: F401 — ensure module is loaded
    from kuaa.library import register_film
    from kuaa.models.clip import openclip as openclip_mod

    # Access the MODULE object via sys.modules — `kuaa.search.aggregate`
    # as an attribute resolves to the `aggregate` function re-exported by the
    # package __init__, not the submodule.
    _csa_mod = sys.modules["kuaa.search.aggregate"]

    real_load = openclip_mod.OpenClipEmbedder.load

    class FakeEmbedder:
        def __init__(self, *args, **kwargs):
            pass

        def encode_text(self, q: str) -> np.ndarray:
            return np.ones(4, dtype=np.float32)

        def encode_image_single(self, path) -> np.ndarray:
            return np.ones(4, dtype=np.float32)

        load = staticmethod(real_load)

    monkeypatch.setattr(_csa_mod, "_get_embedder", lambda cfg: FakeEmbedder())
    monkeypatch.setattr(openclip_mod, "OpenClipEmbedder", FakeEmbedder)

    library_dir = Path(cfg.paths.library_dir)
    for slug, title in [("alpha", "Alpha"), ("beta", "Beta")]:
        register_film(
            library_dir,
            slug=slug,
            title=title,
            year=2026,
            raw_filename=f"{slug}.mp4",
        )
        film_dir = library_dir / slug
        (film_dir / "metadata").mkdir(parents=True, exist_ok=True)
        (film_dir / "embeddings").mkdir(parents=True, exist_ok=True)
        (film_dir / "frames").mkdir(parents=True, exist_ok=True)
        # Minimal scene_descriptions + keyframes_metadata + tag-index + .npy
        (film_dir / "metadata" / "scene_descriptions.json").write_text(
            json.dumps(
                [
                    {"scene_id": 1, "description": "a man on a horse"},
                    {"scene_id": 2, "description": "the year 1959 written on a wall"},
                ]
            )
        )
        (film_dir / "metadata" / "keyframes_metadata.json").write_text(
            json.dumps(
                [
                    {"scene_id": 1, "start_time_s": 1.0, "fps": 24.0},
                    {"scene_id": 2, "start_time_s": 2.0, "fps": 24.0},
                ]
            )
        )
        (film_dir / "metadata" / "scene_tags.json").write_text(
            json.dumps({"outdoor": [1], "interior": [2]})
        )
        (film_dir / "metadata" / "manual_annotations.json").write_text("{}")
        # Stub embeddings: 2 rows of unit ones → cosine vs encode_text() = 4.0
        emb = np.ones((2, 4), dtype=np.float32)
        np.save(film_dir / "embeddings" / "keyframe_embeddings.npy", emb)
        (film_dir / "embeddings" / "index_mapping.json").write_text(
            json.dumps(
                {
                    "model": "stub",
                    "dimension": 4,
                    "total_vectors": 2,
                    "normalized": True,
                    "keyframe_paths": [
                        str(film_dir / "frames" / "1.jpg"),
                        str(film_dir / "frames" / "2.jpg"),
                    ],
                    "scene_ids": [1, 2],
                }
            )
        )
        (film_dir / "frames" / "1.jpg").write_bytes(b"\xff\xd8\xff\xd9")
        (film_dir / "frames" / "2.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    return cfg


def _client() -> TestClient:
    """TestClient with locale pinned to ``en`` — mirrors conftest._make_client.

    The default locale is pt_BR; the en catalog has empty msgstrs that fall
    back to English msgids. Pinning en here keeps snapshots stable against
    pt_BR translation edits during the rest of P1.
    """
    c = TestClient(app)
    c.cookies.set("locale", "en")
    return c


def test_text_search_clip_single_film(small_library):
    client = _client()
    r = client.get(
        "/api/search",
        params={
            "q": "horse",
            "top_k": 5,
            "retriever": "clip",
            "film": "alpha",
        },
    )
    assert r.status_code == 200
    payload = {"status": r.status_code, "html": _slim(r.text)}
    assert_snapshot("p1_search_text__text_clip_single_film", payload)


def test_text_search_hybrid_single_film(small_library):
    client = _client()
    r = client.get(
        "/api/search",
        params={
            "q": "horse",
            "top_k": 5,
            "retriever": "hybrid",
            "film": "alpha",
        },
    )
    assert r.status_code == 200
    payload = {"status": r.status_code, "html": _slim(r.text)}
    assert_snapshot("p1_search_text__text_hybrid_single_film", payload)


def test_text_search_bm25_single_film(small_library):
    client = _client()
    r = client.get(
        "/api/search",
        params={
            "q": "horse",
            "top_k": 5,
            "retriever": "bm25",
            "film": "alpha",
        },
    )
    assert r.status_code == 200
    payload = {"status": r.status_code, "html": _slim(r.text)}
    assert_snapshot("p1_search_text__text_bm25_single_film", payload)


def test_text_search_with_tag_filter(small_library):
    client = _client()
    r = client.get(
        "/api/search",
        params=[
            ("q", "horse"),
            ("top_k", "5"),
            ("retriever", "clip"),
            ("film", "alpha"),
            ("tags", "outdoor"),
        ],
    )
    assert r.status_code == 200
    payload = {"status": r.status_code, "html": _slim(r.text)}
    assert_snapshot("p1_search_text__text_clip_with_tag", payload)


def test_text_search_aggregate_across_films(small_library):
    client = _client()
    r = client.get(
        "/api/search",
        params={
            "q": "horse",
            "top_k": 5,
            "retriever": "clip",
        },
    )  # no ?film= → aggregate path
    assert r.status_code == 200
    payload = {"status": r.status_code, "html": _slim(r.text)}
    assert_snapshot("p1_search_text__text_clip_aggregate", payload)


def test_image_search_single_film(small_library):
    client = _client()
    img_bytes = b"\xff\xd8\xff\xd9"  # tiny stub JPEG
    r = client.post(
        "/api/search/image",
        params={"film": "alpha", "top_k": 5},
        files={"file": ("frame.jpg", img_bytes, "image/jpeg")},
    )
    assert r.status_code == 200
    payload = {"status": r.status_code, "html": _slim(r.text)}
    assert_snapshot("p1_search_image__image_single_film", payload)


def test_no_index_response(tmp_config):
    """Empty library → /api/search returns the no-index UI state."""
    client = _client()
    r = client.get("/api/search", params={"q": "horse", "top_k": 5})
    assert r.status_code == 200
    payload = {"status": r.status_code, "html": _slim(r.text)}
    assert_snapshot("p1_search_edge__no_index_empty_library", payload)


def test_short_query_returns_empty(tmp_config):
    """Query < 2 chars short-circuits without running a search.

    No submit trigger (a bare request) → the body is the (empty) clear-error
    OOB span for ``#search-query-error``, never an ``is-error`` message (U1).
    The dispatcher is not invoked for keystroke noise.
    """
    client = _client()
    r = client.get("/api/search", params={"q": "a", "top_k": 5})
    assert r.status_code == 200
    assert "field-error is-error" not in r.text  # silent on a non-submit short query
    payload = {"status": r.status_code, "html": _slim(r.text)}
    assert_snapshot("p1_search_edge__short_query", payload)


def test_p1_snapshot_uses_shared_helper():
    import tests.test_p1_search_snapshot as mod

    # After migration the bespoke updater is gone and the shared helper is imported.
    assert not hasattr(mod, "_assert_or_update"), "bespoke updater should be removed"
    from tests._snapshot import assert_snapshot  # noqa: F401

    assert getattr(mod, "assert_snapshot", None) is assert_snapshot
