"""Unit tests for ``cinemateca.search.cache``.

These tests pin the relocated index loader + mtime/size cache to its new
home. They are deliberately decoupled from the legacy
``api.services.search`` import path — the service layer re-exports the
same symbols for back-compat, but the canonical home is here.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cinemateca.search.cache import (
    IndexStatus,
    SearchIndex,
    clear_index_cache,
    load_index,
)


@pytest.fixture(autouse=True)
def _isolate():
    clear_index_cache()
    yield
    clear_index_cache()


def _write_minimal_index(emb_path: Path, map_path: Path, n_rows: int = 2) -> None:
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, np.ones((n_rows, 4), dtype=np.float32))
    map_path.write_text(
        json.dumps(
            {
                "model": "stub",
                "dimension": 4,
                "total_vectors": n_rows,
                "normalized": True,
                "keyframe_paths": [f"frames/s{i + 1}.jpg" for i in range(n_rows)],
                "scene_ids": [i + 1 for i in range(n_rows)],
            }
        )
    )


def _register_alpha(library_dir: Path) -> None:
    """Register a minimal ``alpha`` film so ``FilmContext.for_film`` passes."""
    from cinemateca.library import register_film

    register_film(
        library_dir,
        slug="alpha",
        title="Alpha",
        year=2026,
        raw_filename="alpha.mp4",
    )


def test_missing_files_yield_missing_status(tmp_path: Path) -> None:
    from api.services.film_context import FilmContext

    _register_alpha(tmp_path)
    cfg = SimpleNamespace(paths=SimpleNamespace(library_dir=str(tmp_path)))
    ctx = FilmContext.for_film(cfg, "alpha")
    idx = load_index(
        ctx,
        embeddings_filename="keyframe_embeddings.npy",
        mapping_filename="index_mapping.json",
    )
    assert idx.status is IndexStatus.MISSING
    assert idx.ok is False


def test_wellformed_index_loads_ok(tmp_path: Path, monkeypatch) -> None:
    """A well-formed embeddings + mapping yields ``IndexStatus.OK``.

    OpenClipEmbedder is monkey-patched so the test doesn't download CLIP
    weights — we still exercise the real loader's row-count validation
    by handing it the real (numpy, mapping, kf_df) shapes.
    """
    import pandas as pd

    real_open_clip = __import__("cinemateca.models.clip.openclip", fromlist=["OpenClipEmbedder"])

    class FakeEmbedder:
        def __init__(self, *a, **k):
            pass

        @staticmethod
        def load(emb_path: Path, map_path: Path):
            emb = np.load(emb_path)
            mapping = json.loads(Path(map_path).read_text())
            kf_df = pd.DataFrame(
                {
                    "scene_id": mapping["scene_ids"],
                    "keyframe_path": mapping["keyframe_paths"],
                }
            )
            return emb, mapping, kf_df

    monkeypatch.setattr(real_open_clip, "OpenClipEmbedder", FakeEmbedder)

    from api.services.film_context import FilmContext

    _register_alpha(tmp_path)
    cfg = SimpleNamespace(paths=SimpleNamespace(library_dir=str(tmp_path)))
    ctx = FilmContext.for_film(cfg, "alpha")
    _write_minimal_index(
        ctx.embeddings_dir / "keyframe_embeddings.npy",
        ctx.embeddings_dir / "index_mapping.json",
    )
    idx = load_index(
        ctx,
        embeddings_filename="keyframe_embeddings.npy",
        mapping_filename="index_mapping.json",
    )
    assert idx.status is IndexStatus.OK
    assert idx.ok is True
    assert idx.embeddings.shape == (2, 4)
    assert len(idx.kf_df) == 2


def test_clear_cache_drops_entries(tmp_path: Path) -> None:
    """``clear_index_cache()`` empties the module-level cache dict."""
    from cinemateca.search import cache as cache_mod

    cache_mod._index_cache[("x", "/p/e", "/p/m")] = (
        ((0, 0), (0, 0)),
        SearchIndex(status=IndexStatus.MISSING),
    )
    assert cache_mod._index_cache
    clear_index_cache()
    assert not cache_mod._index_cache
