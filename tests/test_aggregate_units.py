"""C1 — per-unit tests for the decomposed aggregate pipeline."""

from __future__ import annotations

from cinemateca.search._aggregate.fusion import fuse_global_rrf


def test_fuse_global_rrf_weights_and_ranks() -> None:
    clip = [(("a", 1), 0.9), (("a", 2), 0.5)]
    bm25 = [(("a", 2), 3.0), (("b", 1), 1.0)]
    fused = fuse_global_rrf([(clip, 0.7), (bm25, 0.3)], k_rrf=60)
    keys = [k for k, _ in fused]
    # scene ("a",2) appears in both lists → highest fused score, ranks first.
    assert keys[0] == ("a", 2)
    assert all(score > 0 for _, score in fused)


def test_fuse_global_rrf_skips_zero_weight_lists() -> None:
    a = [(("a", 1), 1.0)]
    b = [(("b", 1), 1.0)]
    fused = fuse_global_rrf([(a, 1.0), (b, 0.0)], k_rrf=60)
    assert [k for k, _ in fused] == [("a", 1)]
