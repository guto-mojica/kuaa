"""Unit tests for RRF fusion + retriever-mode dispatch.

These tests pin the algorithmic properties: idempotence, pure-CLIP and
pure-BM25 regression behaviour under degenerate weights, and the
degenerate (0, 0) fallback to config defaults.
"""

from __future__ import annotations

import pytest

from cinemateca.retrieval.hybrid import (
    DEFAULT_RRF_K,
    fuse_rrf,
    resolve_weights,
)


def test_fuse_identical_lists_is_stable() -> None:
    a = [(0, 0.9), (1, 0.8), (2, 0.5)]
    b = [(0, 0.9), (1, 0.8), (2, 0.5)]
    fused = fuse_rrf(a, b, sem_w=0.5, bm25_w=0.5)
    assert [sid for sid, _ in fused] == [0, 1, 2]


def test_sem_w_one_bm25_w_zero_matches_clip_order() -> None:
    a = [(0, 0.9), (1, 0.8), (2, 0.5)]
    b = [(2, 1.0), (1, 0.5), (0, 0.1)]
    fused = fuse_rrf(a, b, sem_w=1.0, bm25_w=0.0)
    assert [sid for sid, _ in fused] == [0, 1, 2]


def test_bm25_w_one_sem_w_zero_matches_bm25_order() -> None:
    a = [(0, 0.9), (1, 0.8), (2, 0.5)]
    b = [(2, 1.0), (1, 0.5), (0, 0.1)]
    fused = fuse_rrf(a, b, sem_w=0.0, bm25_w=1.0)
    assert [sid for sid, _ in fused] == [2, 1, 0]


def test_doc_in_one_list_only_gets_partial_score() -> None:
    a = [(0, 0.9), (1, 0.8)]
    b = [(2, 0.7)]
    fused = fuse_rrf(a, b, sem_w=0.5, bm25_w=0.5)
    sids = {sid for sid, _ in fused}
    assert sids == {0, 1, 2}, "All docs from either list must appear"
    by_sid = dict(fused)
    # Doc 0 is rank 1 in A, absent from B → score = 0.5 * 1/(60+1) + 0
    # Doc 2 is rank 1 in B, absent from A → score = 0      + 0.5 * 1/61
    # → same score. The mid-rank doc (1, rank 2 in A) should rank below
    # both. Sort by score desc, top scores tie.
    assert by_sid[0] == pytest.approx(by_sid[2], rel=1e-9)
    assert by_sid[1] < by_sid[0]


def test_mid_weight_interpolates() -> None:
    a = [(0, 0.9), (1, 0.8)]
    b = [(1, 0.9), (0, 0.8)]  # swapped
    fused = fuse_rrf(a, b, sem_w=0.7, bm25_w=0.3)
    by_sid = dict(fused)
    # Doc 0 is rank 1 in A, rank 2 in B → 0.7/61 + 0.3/62
    # Doc 1 is rank 2 in A, rank 1 in B → 0.7/62 + 0.3/61
    # 0.7-weighted side dominates → doc 0 wins.
    assert by_sid[0] > by_sid[1]


def test_resolve_weights_clamps_to_unit_range() -> None:
    assert resolve_weights(sem_w=2.0, bm25_w=-0.5, defaults=(0.7, 0.3)) == (1.0, 0.0)
    assert resolve_weights(sem_w=-1.0, bm25_w=2.0, defaults=(0.7, 0.3)) == (0.0, 1.0)


def test_resolve_weights_degenerate_zero_falls_back() -> None:
    # Both weights == 0 is undefined ordering; fall back to configured defaults.
    assert resolve_weights(sem_w=0.0, bm25_w=0.0, defaults=(0.7, 0.3)) == (0.7, 0.3)


def test_default_rrf_k_constant() -> None:
    assert DEFAULT_RRF_K == 60  # Cormack et al. paper value.
