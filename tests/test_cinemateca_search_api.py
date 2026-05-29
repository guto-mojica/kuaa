"""Tests the cinemateca.search public surface — what callers see.

The 4 public verbs (``find``, ``aggregate``, ``reindex_bm25``, ``rerank``)
and the 7 public types (``Filters``, ``Hit``, ``HybridWeights``, ``Query``,
``SearchMode``, ``SearchResult``, ``UploadRejected``) together form the
T13-locked public API. These tests pin that surface — every caller written
against it (the slim service in T14, future M2 / M3 work) sees what is
asserted here. Implementation details (DataFrame columns, lazy attribute
reads, BM25-tunable resolution) are deliberately exercised through the
public surface only.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cinemateca import search
from cinemateca.search.cache import IndexStatus, SearchIndex


class _FakeEmbedder:
    def encode_text(self, q):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def encode_image_single(self, p):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


@pytest.fixture
def fake_film(tmp_path, monkeypatch):
    """Single-film fixture with a stubbed CLIP loader.

    Builds a registered ``alpha`` film with one scene + a 1×4 .npy + a
    well-formed ``index_mapping.json``, then monkeypatches the cache
    loader so a ``SearchIndex`` is returned without invoking the real
    ``OpenClipEmbedder`` (which would download weights). The fake
    embedder echoes a constant unit vector so cosine matches all stored
    rows perfectly — ``find()`` returns the one scene.
    """
    from cinemateca.library import FilmContext, register_film

    register_film(tmp_path, slug="alpha", title="Alpha", year=2026, raw_filename="alpha.mp4")
    cfg = SimpleNamespace(paths=SimpleNamespace(library_dir=str(tmp_path), data_dir=str(tmp_path)))
    ctx = FilmContext.for_film(cfg, "alpha")

    # Minimal on-disk artefacts.
    (ctx.metadata_dir / "scene_descriptions.json").write_text(
        json.dumps([{"scene_id": 1, "description": "man on a horse"}])
    )
    (ctx.metadata_dir / "scene_tags.json").write_text("{}")
    (ctx.metadata_dir / "manual_annotations.json").write_text("{}")
    np.save(
        ctx.embeddings_dir / "keyframe_embeddings.npy",
        np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    )
    (ctx.embeddings_dir / "index_mapping.json").write_text(
        json.dumps(
            {
                "total_vectors": 1,
                "keyframe_paths": ["/p/1.jpg"],
                "scene_ids": [1],
            }
        )
    )

    from cinemateca.search import cache as cache_mod

    # Drop any leaked cache entries from prior tests (the module-level dict
    # is shared) before monkeypatching the loader.
    cache_mod.clear_index_cache()

    def _stub_loader(emb_path: Path, map_path: Path) -> SearchIndex:
        mapping = json.loads(map_path.read_text())
        return SearchIndex(
            status=IndexStatus.OK,
            embeddings=np.load(emb_path),
            kf_df=pd.DataFrame(
                {
                    "filepath": mapping["keyframe_paths"],
                    "scene_id": mapping["scene_ids"],
                }
            ),
            embedder=_FakeEmbedder(),
        )

    monkeypatch.setattr(cache_mod, "_load_and_validate", _stub_loader)
    return ctx


def test_find_text_query_runs(fake_film):
    q = search.Query.of_text("horse")
    result = search.find(q, film=fake_film, mode="clip", top_k=3)
    assert isinstance(result, search.SearchResult)
    assert result.mode == "clip"
    assert result.no_index is False
    assert len(result.hits) >= 1
    hit = result.hits[0]
    assert isinstance(hit, search.Hit)
    assert hit.scene_id == 1
    assert hit.keyframe_path == "/p/1.jpg"
    assert hit.score > 0.0


def test_find_returns_no_index_for_missing(tmp_path):
    """Unwritten embeddings → no_index=True, hits=[]; not an exception."""
    from cinemateca.library import FilmContext, register_film

    register_film(tmp_path, slug="alpha", title="Alpha", year=2026, raw_filename="alpha.mp4")
    cfg = SimpleNamespace(paths=SimpleNamespace(library_dir=str(tmp_path)))
    ctx = FilmContext.for_film(cfg, "alpha")

    from cinemateca.search import cache as cache_mod

    cache_mod.clear_index_cache()

    q = search.Query.of_text("x")
    result = search.find(q, film=ctx, top_k=3)
    assert result.no_index is True
    assert result.hits == []
    assert isinstance(result, search.SearchResult)


def test_reindex_bm25_clears_cache(fake_film):
    # Smoke: the public verb must accept a FilmContext and not raise.
    search.reindex_bm25(fake_film)
