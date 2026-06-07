"""Test the per-FilmContext BM25 loader, including cache invalidation
across all three tag-source files."""

from __future__ import annotations

import json
import time
from pathlib import Path

from api.services.search import _get_bm25_index_for_ctx
from cinemateca.library import FilmContext


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


def test_file_stamp_uses_nanosecond_resolution(tmp_path: Path) -> None:
    """``_file_stamp`` must distinguish writes that differ only at sub-second
    resolution — float ``st_mtime`` collapses two same-second writes into the
    same stamp, allowing a stale BM25 cache to survive a content change.
    """
    import os

    from api.services.search import _file_stamp

    p = tmp_path / "f.json"
    p.write_text("x")
    # Two mtimes differing by exactly 1 nanosecond. float st_mtime cannot
    # represent that difference at any plausible epoch (1.7e9 + 1e-9 ==
    # 1.7e9 in IEEE-754 double), so a float-stamp implementation returns
    # identical tuples; an int-ns implementation returns two distinct ints.
    os.utime(p, ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))
    stamp1 = _file_stamp(p)
    os.utime(p, ns=(1_700_000_000_000_000_001, 1_700_000_000_000_000_001))
    stamp2 = _file_stamp(p)
    assert stamp1 != stamp2, (
        f"1-ns mtime difference must yield a different file stamp; "
        f"got {stamp1} == {stamp2} (st_mtime float resolution is not enough)"
    )


def test_clear_index_cache_also_clears_bm25_lru_cache(tmp_path: Path) -> None:
    """``clear_index_cache()`` must invalidate the BM25 lru_cache too.

    Without this, tests/conftest.py's per-test ``clear_index_cache()``
    hook leaves the module-level ``@lru_cache(maxsize=32)`` populated;
    a fresh test whose ``tmp_path`` (or, more subtly, whose mocked
    on-disk shape) happens to collide with a previous test's cache key
    reuses the stale BM25 index — false-positive pass or wrong-corpus
    failure.

    We don't synthesise a collision (hard to make deterministic across
    pytest invocations); we directly verify the cache-clear hook does
    what it advertises by checking ``cache_info().currsize`` drops to 0.
    """
    from api.services.search import _cached_bm25_index, clear_index_cache

    ctx = _make_ctx(tmp_path)
    _ = _get_bm25_index_for_ctx(ctx)
    assert _cached_bm25_index.cache_info().currsize >= 1, (
        "sanity check: BM25 loader must populate the lru_cache"
    )
    clear_index_cache()
    assert _cached_bm25_index.cache_info().currsize == 0, (
        "clear_index_cache() must also flush _cached_bm25_index — "
        "leaving it populated breaks per-test isolation in conftest.py"
    )


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


# ── T7: new tests for the core ``cinemateca.search.bm25`` module ─────────────
#
# These exercise the relocated module directly (rather than via the
# ``api.services.search`` re-export shim) so the boundary holds even if
# the shim is later trimmed further. The 3-doc corpora here mirror the
# fixture above — rank_bm25 IDF on a 2-doc corpus with df=1 collapses
# every query score to 0 (BM25Index.query drops zero-score hits), so a
# 2-doc plan-as-written would assert against an empty list.


def _write_three_doc_metadata(metadata_dir: Path) -> None:
    """Three-document corpus — enough for non-degenerate BM25 IDF.

    With ≥3 docs and df=1 for a query term, the IDF is positive and
    :meth:`BM25Index.query` returns the matching scene.
    """
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "scene_descriptions.json").write_text(
        json.dumps(
            [
                {"scene_id": 1, "description": "a man on a horse outdoor"},
                {"scene_id": 2, "description": "interior shot of a house"},
                {"scene_id": 3, "description": "a red car on the road"},
            ]
        )
    )
    (metadata_dir / "scene_tags.json").write_text(json.dumps({}))
    (metadata_dir / "manual_annotations.json").write_text("{}")


def test_bm25_index_for_dir_builds_corpus(tmp_path: Path) -> None:
    """``bm25_index_for_dir`` (the pure-Path loader) builds a queryable index."""
    from cinemateca.search.bm25 import bm25_index_for_dir, clear_bm25_cache

    clear_bm25_cache()
    _write_three_doc_metadata(tmp_path)
    idx = bm25_index_for_dir(
        metadata_dir=tmp_path,
        stopwords_lang=None,
        k1=1.5,
        b=0.75,
    )
    hits = idx.query("horse", top_k=5)
    assert len(hits) == 1
    assert hits[0][0] == 1
    clear_bm25_cache()


def test_bm25_index_for_dir_empty_corpus_returns_no_hits(tmp_path: Path) -> None:
    """An empty descriptions + empty tags corpus yields a no-match index."""
    from cinemateca.search.bm25 import bm25_index_for_dir, clear_bm25_cache

    clear_bm25_cache()
    (tmp_path / "scene_descriptions.json").write_text("[]")
    (tmp_path / "scene_tags.json").write_text("{}")
    (tmp_path / "manual_annotations.json").write_text("{}")
    idx = bm25_index_for_dir(
        metadata_dir=tmp_path,
        stopwords_lang=None,
        k1=1.5,
        b=0.75,
    )
    assert idx.query("anything", top_k=5) == []
    clear_bm25_cache()


def test_reindex_bm25_clears_slot(tmp_path: Path) -> None:
    """``reindex_bm25(ctx)`` must invalidate the BM25 cache.

    After reindex, the next load reads fresh on-disk descriptions —
    proven by querying a token that only exists in the new corpus.
    """
    from types import SimpleNamespace

    from cinemateca.search.bm25 import (
        bm25_index_for_dir,
        clear_bm25_cache,
        reindex_bm25,
    )

    clear_bm25_cache()
    _write_three_doc_metadata(tmp_path)
    ctx = SimpleNamespace(metadata_dir=tmp_path, slug="alpha")
    _ = bm25_index_for_dir(metadata_dir=tmp_path, stopwords_lang=None, k1=1.5, b=0.75)
    reindex_bm25(ctx)
    # Replace the corpus and verify the new tokens are queryable.
    (tmp_path / "scene_descriptions.json").write_text(
        json.dumps(
            [
                {"scene_id": 99, "description": "fresh new content"},
                {"scene_id": 100, "description": "another fresh segment"},
                {"scene_id": 101, "description": "third fresh keyframe"},
            ]
        )
    )
    idx2 = bm25_index_for_dir(metadata_dir=tmp_path, stopwords_lang=None, k1=1.5, b=0.75)
    hits = idx2.query("content", top_k=5)
    assert hits and hits[0][0] == 99
    clear_bm25_cache()
