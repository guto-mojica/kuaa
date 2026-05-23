"""Test the per-FilmContext BM25 loader, including cache invalidation
across all three tag-source files."""

from __future__ import annotations

import json
import time
from pathlib import Path

from api.services.film_context import FilmContext
from api.services.search import _get_bm25_index_for_ctx


def _make_ctx(tmp_path: Path) -> FilmContext:
    """Build a minimal FilmContext for testing.

    FilmContext is a frozen dataclass with six required fields. None of
    them other than ``metadata_dir`` and ``slug`` are exercised by the
    BM25 loader — but the dataclass refuses to instantiate without all
    six, so we populate them with sensible per-film-layout paths.
    """
    film_dir = tmp_path / "library" / "demo"
    metadata_dir = film_dir / "metadata"
    metadata_dir.mkdir(parents=True)
    # Need ≥3 docs for BM25 idf on a term-in-one-doc to be positive
    # (with 2 docs and df=1, idf = log((2-1+0.5)/(1+0.5)) = log(1) = 0,
    # and BM25Index drops zero-score docs as no-match).
    (metadata_dir / "scene_descriptions.json").write_text(
        json.dumps(
            [
                {"scene_id": 0, "description": "menina chorando na chuva"},
                {"scene_id": 1, "description": "homem caminhando na rua"},
                {"scene_id": 2, "description": "carro vermelho na estrada"},
            ]
        )
    )
    (metadata_dir / "scene_tags.json").write_text(json.dumps({}))
    (metadata_dir / "manual_annotations.json").write_text(json.dumps({}))

    return FilmContext(
        slug="demo",
        raw_path=film_dir / "raw",
        data_dir=tmp_path.resolve(),
        metadata_dir=metadata_dir,
        frames_dir=film_dir / "frames",
        embeddings_dir=film_dir / "embeddings",
    )


def test_loader_returns_built_index(tmp_path: Path) -> None:
    ctx = _make_ctx(tmp_path)
    idx = _get_bm25_index_for_ctx(ctx)
    assert idx is not None
    hits = idx.query("menina", top_k=5)
    assert hits[0][0] == 0


def test_loader_cache_invalidates_on_descriptions_change(tmp_path: Path) -> None:
    ctx = _make_ctx(tmp_path)
    a = _get_bm25_index_for_ctx(ctx)
    b = _get_bm25_index_for_ctx(ctx)
    assert a is b, "Cache hit must return same object"

    time.sleep(0.01)
    (ctx.metadata_dir / "scene_descriptions.json").write_text(
        json.dumps(
            [
                {"scene_id": 0, "description": "menina chorando na chuva"},
                {"scene_id": 1, "description": "homem caminhando na rua"},
                {"scene_id": 2, "description": "carro vermelho na estrada"},
                {"scene_id": 3, "description": "casa antiga no campo"},
            ]
        )
    )
    c = _get_bm25_index_for_ctx(ctx)
    assert c is not a, "mtime+size change on descriptions must rebuild"


def test_loader_cache_invalidates_on_manual_annotations_change(tmp_path: Path) -> None:
    """A write to manual_annotations.json must also bust the cache."""
    ctx = _make_ctx(tmp_path)
    a = _get_bm25_index_for_ctx(ctx)
    time.sleep(0.01)
    (ctx.metadata_dir / "manual_annotations.json").write_text(
        json.dumps(
            {
                "exterior": ["0"],
            }
        )
    )
    b = _get_bm25_index_for_ctx(ctx)
    assert b is not a, "manual_annotations.json write must invalidate the BM25 cache"
