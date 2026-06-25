"""HTTP-layer tests for the M2 hybrid-search dispatch.

These tests confirm ``/api/search`` routes correctly through the three
retrieval modes (``clip`` / ``bm25`` / ``hybrid``), validates inputs
(weight clamps, unknown mode, degenerate zero-weights), and that the
legacy default flips to ``hybrid``.

The route's INFO log line is a secondary pin: ``api_search`` emits
``"retriever=<mode> sw=… bw=…"`` once per request, which lets us
distinguish "the dispatcher chose hybrid by default" from "FastAPI
silently dropped an unknown query param" — the latter would never log.

The dispatch tests use a tiny real on-disk per-film index plus
``scene_descriptions.json`` BM25 corpus. The CLIP encoder itself is still
stubbed, but the route now reaches the actual CLIP/BM25/hybrid branches
instead of stopping at an empty-library no-index response.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pytest

from kuaa.library import register_film

SEARCH_NO_INDEX = "No search index found. Run the pipeline with the Embeddings step first."
SLUG = "route_film"


def _seed_indexed_film(cfg) -> None:
    """Create one real per-film CLIP index plus BM25 source metadata.

    The fixture intentionally makes the semantic and lexical winners differ:
    scene 1 has the best CLIP vector for the stub query, while scene 0 is the
    only description containing "menina". Route tests can therefore prove
    that ``retriever=clip`` and ``retriever=bm25`` are not just accepted; they
    dispatch to different retrieval branches.
    """
    library_dir = Path(cfg.paths.library_dir)
    data_dir = Path(cfg.paths.data_dir)
    register_film(
        library_dir,
        slug=SLUG,
        title="Route Film",
        year=1959,
        raw_filename=f"{SLUG}.mp4",
    )
    film_dir = library_dir / SLUG
    (film_dir / "raw").mkdir(parents=True, exist_ok=True)
    (film_dir / "raw" / f"{SLUG}.mp4").write_bytes(b"")
    meta_dir = film_dir / "metadata"
    emb_dir = film_dir / "embeddings"
    meta_dir.mkdir(parents=True, exist_ok=True)
    emb_dir.mkdir(parents=True, exist_ok=True)

    keyframe_paths: list[str] = []
    keyframes: list[dict] = []
    for scene_id in (0, 1, 2):
        frame = data_dir / "library" / SLUG / "frames" / "keyframes" / f"{scene_id}.jpg"
        frame.parent.mkdir(parents=True, exist_ok=True)
        frame.write_bytes(b"")
        keyframe_paths.append(str(frame))
        keyframes.append(
            {
                "scene_id": scene_id,
                "filepath": str(frame),
                "start_time_s": float(scene_id),
            }
        )
    (meta_dir / "keyframes_metadata.json").write_text(json.dumps(keyframes))
    (meta_dir / "scene_descriptions.json").write_text(
        json.dumps(
            [
                {"scene_id": 0, "description": "menina chorando na chuva"},
                {"scene_id": 1, "description": "homem caminhando na rua"},
                {"scene_id": 2, "description": "carro vermelho na estrada"},
            ]
        )
    )
    (meta_dir / "scene_tags.json").write_text(json.dumps({"outdoor": [0, 1, 2]}))

    vectors = np.array(
        [
            [0.0, 1.0],  # scene 0: lexical winner, semantic loser
            [1.0, 0.0],  # scene 1: semantic winner
            [0.5, 0.5],
        ],
        dtype=np.float32,
    )
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    np.save(emb_dir / cfg.embeddings.filename, vectors)
    (emb_dir / cfg.embeddings.mapping_filename).write_text(
        json.dumps(
            {
                "model": "stub",
                "dimension": 2,
                "total_vectors": 3,
                "normalized": True,
                "keyframe_paths": keyframe_paths,
                "scene_ids": [0, 1, 2],
                "keyframe_ids": [0, 1, 2],
            }
        )
    )


@pytest.fixture()
def indexed_search_client(tmp_config, monkeypatch, client):
    _seed_indexed_film(tmp_config)

    import api.services.search as search_service
    import kuaa.models.clip.openclip as openclip

    real_load = openclip.OpenClipEmbedder.load

    class StubEmbedder:
        load = staticmethod(real_load)

        def encode_text(self, q: str) -> np.ndarray:
            return np.array([1.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(openclip, "OpenClipEmbedder", StubEmbedder)
    monkeypatch.setattr(search_service, "_get_embedder", lambda cfg: StubEmbedder())
    search_service.clear_index_cache()
    return client


def _scene_ids(html: str) -> list[int]:
    return [int(m) for m in re.findall(r'data-scene-id="(\d+)"', html)]


def test_search_route_clip_uses_semantic_index(indexed_search_client) -> None:
    """``retriever=clip`` short-circuits to pure CLIP — the regression-pin path.

    ``reranker_enabled=false`` isolates the retrieval stage under test: on a GPU
    box the reranker default is ``auto``-on, and the cross-encoder would reorder
    these results by description relevance. Rerank has its own coverage in
    ``test_search_rerank`` / ``test_search_service_with_reranker``.
    """
    resp = indexed_search_client.get(
        "/api/search",
        params={
            "q": "menina",
            "retriever": "clip",
            "film": SLUG,
            "top_k": 1,
            "reranker_enabled": "false",
        },
    )
    assert resp.status_code == 200
    assert SEARCH_NO_INDEX not in resp.text
    assert _scene_ids(resp.text) == [1]


def test_search_route_bm25_uses_real_description_corpus(indexed_search_client) -> None:
    resp = indexed_search_client.get(
        "/api/search",
        params={
            "q": "menina",
            "retriever": "bm25",
            "film": SLUG,
            "top_k": 3,
            "reranker_enabled": "false",  # isolate retrieval ordering (see clip test)
        },
    )
    assert resp.status_code == 200
    assert SEARCH_NO_INDEX not in resp.text
    assert _scene_ids(resp.text) == [0]


def test_search_route_hybrid_with_weights_changes_result_order(indexed_search_client) -> None:
    resp = indexed_search_client.get(
        "/api/search",
        params={
            "q": "menina",
            "retriever": "hybrid",
            "sem_w": 1.0,
            "bm25_w": 0.0,
            "film": SLUG,
            "top_k": 1,
            "reranker_enabled": "false",  # isolate retrieval ordering (see clip test)
        },
    )
    assert resp.status_code == 200
    assert _scene_ids(resp.text) == [1]


def test_search_route_default_retriever_is_hybrid(client, caplog) -> None:
    """No ``retriever`` param ⇒ hybrid. Verified by the route's INFO log line."""
    with caplog.at_level(logging.INFO, logger="api.routes.search"):
        resp = client.get("/api/search", params={"q": "menina"})
    assert resp.status_code == 200
    # ``api_search`` emits the canonical mode/weights line every request;
    # the substring ``retriever=hybrid`` is the regression pin that hybrid
    # is the M2 default (and that FastAPI did NOT silently drop the param).
    assert any("retriever=hybrid" in r.getMessage() for r in caplog.records)


def test_search_route_unknown_retriever_falls_back_to_default(client, caplog) -> None:
    """An unknown retriever value warns + falls back to ``hybrid``."""
    with caplog.at_level(logging.WARNING, logger="api.routes.search"):
        resp = client.get("/api/search", params={"q": "menina", "retriever": "foobar"})
    assert resp.status_code == 200
    assert any("unknown retriever" in r.getMessage().lower() for r in caplog.records)


def test_search_route_clamps_out_of_range_weights(indexed_search_client, caplog) -> None:
    """``sem_w=2`` and ``bm25_w=-0.5`` get clamped to ``(1.0, 0.0)`` (still valid)."""
    with caplog.at_level(logging.INFO, logger="api.routes.search"):
        resp = indexed_search_client.get(
            "/api/search",
            params={
                "q": "menina",
                "retriever": "hybrid",
                "sem_w": 2.0,
                "bm25_w": -0.5,
                "film": SLUG,
                "top_k": 1,
                "reranker_enabled": "false",  # isolate retrieval ordering (see clip test)
            },
        )
    assert resp.status_code == 200
    assert _scene_ids(resp.text) == [1]
    assert any("sw=1.000 bw=0.000" in r.getMessage() for r in caplog.records)


def test_search_route_degenerate_zero_weights_falls_back_to_defaults(
    indexed_search_client, caplog
) -> None:
    """``sem_w=0`` ∧ ``bm25_w=0`` falls back to config defaults (0.70/0.30)."""
    with caplog.at_level(logging.INFO, logger="api.routes.search"):
        resp = indexed_search_client.get(
            "/api/search",
            params={
                "q": "menina",
                "retriever": "hybrid",
                "sem_w": 0.0,
                "bm25_w": 0.0,
                "film": SLUG,
                "top_k": 1,
                "reranker_enabled": "false",  # isolate retrieval ordering (see clip test)
            },
        )
    assert resp.status_code == 200
    assert _scene_ids(resp.text) == [0]
    assert any("sw=0.700 bw=0.300" in r.getMessage() for r in caplog.records)


def test_search_offset_paginates_distinct_results(indexed_search_client) -> None:
    """#1: ``offset`` must page into deeper results, not slice an already
    ``top_k``-truncated list to empty. Page 2 returns a distinct, non-empty
    scene — proving the first stage fetched ``top_k + offset`` candidates.
    """

    def _page(offset: int) -> list[int]:
        r = indexed_search_client.get(
            "/api/search",
            params={
                "q": "menina",
                "retriever": "clip",
                "film": SLUG,
                "top_k": 1,
                "offset": offset,
            },
        )
        assert r.status_code == 200
        return _scene_ids(r.text)

    page0 = _page(0)
    page1 = _page(1)
    assert len(page0) == 1 and len(page1) == 1, (page0, page1)  # both pages non-empty
    assert page0 != page1  # offset reached a deeper result, not empty / a dup
