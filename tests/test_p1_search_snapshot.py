"""Hermetic characterization snapshots for /api/search.

These tests fix the HTML structure /api/search produces under the
``tmp_config`` fixture (no real CLIP model, no real library) BEFORE any
P1 code moves. Every subsequent commit on this branch must keep them
passing — that is the contract that the refactor is behavior-preserving.

Regenerate with ``UPDATE_P1_SNAPSHOT=1 uv run pytest tests/test_p1_search_snapshot.py``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.server import app

SNAPSHOTS_DIR = Path(__file__).parent / "fixtures" / "refactor_snapshots"
UPDATE = bool(os.environ.get("UPDATE_P1_SNAPSHOT"))


def _slim(html: str) -> str:
    """Reduce HTML to a structural fingerprint stable across whitespace edits."""
    import re

    return re.sub(r"\s+", " ", html).strip()


def _load_or_init(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _save(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


@pytest.fixture
def small_library(tmp_config, monkeypatch):
    """A two-film hermetic library with stub CLIP + BM25 corpus.

    Mirrors tests/test_multi_film_search.py's fixture style: real on-disk
    JSON + .npy stubs + monkeypatched embedder.

    The CLIP embedder is stubbed via two surfaces:

      * ``api.services.search._get_embedder`` — used by ``aggregate_search``
        (cross-film path) to encode the query text once.
      * ``cinemateca.models.clip.openclip.OpenClipEmbedder`` — replaced with
        a fake that keeps the real ``.load`` staticmethod (the on-disk
        mapping is well-formed) but stubs the constructor + ``encode_*``
        instance methods so ``_load_and_validate`` builds a SearchIndex
        whose embedder produces 4-dim vectors matching the stub .npy.
    """
    cfg = tmp_config
    from api.services import search as svc
    from cinemateca.library import register_film
    from cinemateca.models.clip import openclip as openclip_mod

    real_load = openclip_mod.OpenClipEmbedder.load

    class FakeEmbedder:
        def __init__(self, *args, **kwargs):
            pass

        def encode_text(self, q: str) -> np.ndarray:
            return np.ones(4, dtype=np.float32)

        def encode_image_single(self, path) -> np.ndarray:
            return np.ones(4, dtype=np.float32)

        load = staticmethod(real_load)

    monkeypatch.setattr(svc, "_get_embedder", lambda cfg: FakeEmbedder())
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


def _assert_or_update(snapshot_path: Path, key: str, payload: dict) -> None:
    data = _load_or_init(snapshot_path)
    if UPDATE:
        data[key] = payload
        _save(snapshot_path, data)
        return
    assert key in data, (
        f"No snapshot for {key!r} in {snapshot_path.name}. "
        f"Run with UPDATE_P1_SNAPSHOT=1 to capture."
    )
    assert data[key] == payload, (
        f"Snapshot drift for {key!r} in {snapshot_path.name}. " f"Investigate before regenerating."
    )


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
    _assert_or_update(
        SNAPSHOTS_DIR / "p1_search_text.json",
        "text_clip_single_film",
        payload,
    )


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
    _assert_or_update(
        SNAPSHOTS_DIR / "p1_search_text.json",
        "text_hybrid_single_film",
        payload,
    )


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
    _assert_or_update(
        SNAPSHOTS_DIR / "p1_search_text.json",
        "text_bm25_single_film",
        payload,
    )


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
    _assert_or_update(
        SNAPSHOTS_DIR / "p1_search_text.json",
        "text_clip_with_tag",
        payload,
    )


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
    _assert_or_update(
        SNAPSHOTS_DIR / "p1_search_text.json",
        "text_clip_aggregate",
        payload,
    )


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
    _assert_or_update(
        SNAPSHOTS_DIR / "p1_search_image.json",
        "image_single_film",
        payload,
    )


def test_no_index_response(tmp_config):
    """Empty library → /api/search returns the no-index UI state."""
    client = _client()
    r = client.get("/api/search", params={"q": "horse", "top_k": 5})
    assert r.status_code == 200
    payload = {"status": r.status_code, "html": _slim(r.text)}
    _assert_or_update(
        SNAPSHOTS_DIR / "p1_search_edge.json",
        "no_index_empty_library",
        payload,
    )


def test_short_query_returns_empty(tmp_config):
    """Query shorter than 2 chars short-circuits to an empty HTMLResponse."""
    client = _client()
    r = client.get("/api/search", params={"q": "a", "top_k": 5})
    assert r.status_code == 200
    payload = {"status": r.status_code, "html": _slim(r.text)}
    _assert_or_update(
        SNAPSHOTS_DIR / "p1_search_edge.json",
        "short_query",
        payload,
    )
