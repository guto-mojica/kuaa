"""VectorIndex Protocol + backends + SemanticSearch delegation (Seam 2).

Hermetic: no GPU, no heavy model deps, no LanceDB required (the lancedb
backend is only *constructed* here — it connects lazily, so asserting its
type needs no install). The numpy backend is checked for byte-identical
parity against the legacy inline ``argsort`` path.
"""

from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.smoke


def _unit_rows(n: int, d: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal((n, d)).astype("float32")
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    rows = pd.DataFrame(
        {"scene_id": list(range(1, n + 1)), "filepath": [f"kf_{i}.jpg" for i in range(n)]}
    )
    return emb, rows


# ─── Protocol ─────────────────────────────────────────────────────────────────


def test_vector_index_protocol_runtime_checkable():
    from kuaa.retrieval.vector_index.base import VectorIndex
    from kuaa.retrieval.vector_index.numpy_bruteforce import NumpyBruteForceIndex

    assert getattr(VectorIndex, "_is_runtime_protocol", False)
    assert isinstance(NumpyBruteForceIndex(), VectorIndex)

    class _Bad:
        def add(self, embeddings, rows):
            pass

    assert isinstance(_Bad(), VectorIndex) is False  # missing search/knn


# ─── numpy_bruteforce parity ──────────────────────────────────────────────────


def test_numpy_bruteforce_matches_legacy_argsort():
    from kuaa.retrieval.vector_index.numpy_bruteforce import NumpyBruteForceIndex

    emb, rows = _unit_rows(25, 8, seed=1)
    q = emb[7]

    idx = NumpyBruteForceIndex()
    idx.add(emb, rows)
    hits = idx.search(q, 5)

    # Legacy inline computation.
    sims = (emb @ q).flatten()
    order = np.argsort(sims)[::-1][:5]

    assert list(hits["scene_id"]) == [int(rows.iloc[i]["scene_id"]) for i in order]
    np.testing.assert_allclose(hits["score"].to_numpy(), sims[order].astype(float))


def test_numpy_bruteforce_k_none_returns_all_sorted():
    from kuaa.retrieval.vector_index.numpy_bruteforce import NumpyBruteForceIndex

    emb, rows = _unit_rows(10, 4, seed=2)
    idx = NumpyBruteForceIndex()
    idx.add(emb, rows)
    hits = idx.search(emb[0], None)
    assert len(hits) == 10
    scores = hits["score"].to_numpy()
    assert np.all(np.diff(scores) <= 1e-6)  # descending


def test_numpy_bruteforce_film_slug_filter_and_accrual():
    from kuaa.retrieval.vector_index.numpy_bruteforce import NumpyBruteForceIndex

    emb_a, _ = _unit_rows(3, 4, seed=3)
    emb_b, _ = _unit_rows(3, 4, seed=4)
    rows_a = pd.DataFrame({"scene_id": [1, 2, 3], "film_slug": ["a"] * 3})
    rows_b = pd.DataFrame({"scene_id": [1, 2, 3], "film_slug": ["b"] * 3})

    idx = NumpyBruteForceIndex()
    idx.add(emb_a, rows_a)
    idx.add(emb_b, rows_b)  # multi-add accrual

    hits = idx.search(emb_b[0], 10, film_slug="b")
    assert set(hits["film_slug"]) == {"b"}
    assert len(hits) == 3


def test_numpy_bruteforce_delete_removes_only_matching_film():
    from kuaa.retrieval.vector_index.numpy_bruteforce import NumpyBruteForceIndex

    emb_a, _ = _unit_rows(3, 4, seed=3)
    emb_b, _ = _unit_rows(2, 4, seed=4)
    rows_a = pd.DataFrame({"scene_id": [1, 2, 3], "film_slug": ["a"] * 3})
    rows_b = pd.DataFrame({"scene_id": [1, 2], "film_slug": ["b"] * 2})

    idx = NumpyBruteForceIndex()
    idx.add(emb_a, rows_a)
    idx.add(emb_b, rows_b)

    idx.delete("a")
    hits = idx.search(emb_b[0], None)
    assert set(hits["film_slug"]) == {"b"}
    assert len(hits) == 2


def test_numpy_bruteforce_delete_then_add_is_idempotent():
    """The reindex-vectors use case: delete-then-add must not accumulate
    duplicate rows across repeated runs for the same film."""
    from kuaa.retrieval.vector_index.numpy_bruteforce import NumpyBruteForceIndex

    emb, rows = _unit_rows(4, 4, seed=5)
    rows = rows.assign(film_slug="x")

    idx = NumpyBruteForceIndex()
    for _ in range(3):  # simulate re-running the backfill 3 times
        idx.delete("x")
        idx.add(emb, rows)

    hits = idx.search(emb[0], None)
    assert len(hits) == 4


def test_numpy_bruteforce_row_mismatch_not_validated_crashes_at_search():
    """No eager validation in ``add`` — matches the AI core's historical
    contract (see kuaa.search.cache: shape validation lives in the service
    layer, not here). A mismatched index is accepted silently and only
    crashes later, at out-of-bounds positional indexing in ``search``.
    """
    from kuaa.retrieval.vector_index.numpy_bruteforce import NumpyBruteForceIndex

    emb, rows = _unit_rows(5, 4)
    idx = NumpyBruteForceIndex()
    idx.add(emb, rows.head(3))  # no raise — mismatch silently accepted

    with pytest.raises(IndexError):
        idx.search(emb[0], None)  # order spans all 5 rows; rows only has 3


# ─── factory + config ─────────────────────────────────────────────────────────


def _cfg(backend: str):
    sn = types.SimpleNamespace
    return sn(search=sn(index_backend=backend), paths=sn(library_dir="/tmp/kuaa-lib"))


def test_factory_returns_numpy_default():
    from kuaa.retrieval.vector_index import get_vector_index
    from kuaa.retrieval.vector_index.numpy_bruteforce import NumpyBruteForceIndex

    assert isinstance(get_vector_index(_cfg("numpy_bruteforce")), NumpyBruteForceIndex)


def test_factory_returns_lancedb_without_install():
    # LanceDBIndex constructs without importing lancedb (connection is lazy),
    # so the factory is exercisable even when the scale extra is absent.
    from kuaa.retrieval.vector_index import get_vector_index
    from kuaa.retrieval.vector_index.lancedb_index import LanceDBIndex

    idx = get_vector_index(_cfg("lancedb"))
    assert isinstance(idx, LanceDBIndex)
    assert idx.uri.endswith("vector_index.lancedb")


def test_lancedb_delete_then_add_is_idempotent(tmp_path):
    """Real on-disk LanceDB round-trip (skipped without the 'scale' extra).

    Mirrors ``kuaa library reindex-vectors``: delete-then-add for the same
    film_slug, run repeatedly, must not accumulate duplicate rows.
    """
    pytest.importorskip("lancedb")
    from kuaa.retrieval.vector_index.lancedb_index import LanceDBIndex

    emb, rows = _unit_rows(4, 4, seed=6)
    rows = rows.assign(film_slug="x")

    idx = LanceDBIndex(tmp_path / "vector_index.lancedb")
    for _ in range(3):  # simulate re-running the backfill 3 times
        idx.delete("x")
        idx.add(emb, rows)

    hits = idx.search(emb[0], None)
    assert len(hits) == 4


def test_factory_unknown_backend_raises():
    from kuaa.retrieval.vector_index import get_vector_index

    with pytest.raises(ValueError):
        get_vector_index(_cfg("redis"))


def test_config_default_index_backend_is_numpy():
    from kuaa.config import load_config

    cfg = load_config(project_root=".", ensure_dirs=False)
    assert cfg.search.index_backend == "numpy_bruteforce"


def test_config_index_backend_literal_enforced(tmp_path):
    import yaml

    from kuaa.config import load_config
    from kuaa.errors import ConfigError

    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"search": {"index_backend": "redis"}}), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(bad, project_root=".", ensure_dirs=False)


# ─── SemanticSearch delegation (byte-identical) ───────────────────────────────


class _FakeEmbedder:
    def __init__(self, vec):
        self._vec = np.asarray(vec, dtype="float32")

    def encode_text(self, query):
        return self._vec

    def encode_image_single(self, image_path):
        return self._vec


def test_semantic_search_by_text_byte_identical():
    from kuaa.embeddings import SemanticSearch

    emb = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype="float32")
    df = pd.DataFrame({"scene_id": [1, 2, 3], "filepath": ["a.jpg", "b.jpg", "c.jpg"]})
    q = np.array([1.0, 0.0], dtype="float32")

    out = SemanticSearch(emb, df, _FakeEmbedder(q)).by_text("x", top_k=3)

    sims = (emb @ q).flatten()
    order = np.argsort(sims)[::-1][:3]
    assert list(out["rank"]) == [1, 2, 3]
    assert list(out["scene_id"]) == [int(df.iloc[i]["scene_id"]) for i in order]
    np.testing.assert_allclose(out["similarity"].to_numpy(), sims[order].astype(float))


def test_semantic_search_by_image_excludes_self():
    from kuaa.embeddings import SemanticSearch

    emb = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype="float32")
    df = pd.DataFrame({"scene_id": [1, 2, 3], "filepath": ["a.jpg", "b.jpg", "c.jpg"]})
    # Query equals row b → b would rank #1; exclude_self must drop it.
    out = SemanticSearch(emb, df, _FakeEmbedder(emb[1])).by_image(
        "b.jpg", top_k=2, exclude_self=True
    )
    assert "b.jpg" not in list(out["filepath"])
    assert len(out) == 2
