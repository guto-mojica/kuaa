"""Verify the new bm25 sub-config loads with sane defaults."""

from __future__ import annotations

from kuaa.config import load_config


def test_bm25_defaults_loaded() -> None:
    cfg = load_config()
    bm25 = getattr(cfg.search, "bm25", None) or cfg.search.__dict__.get("bm25")
    assert bm25 is not None, "search.bm25 sub-config not loaded"
    k1 = bm25.k1 if hasattr(bm25, "k1") else bm25["k1"]
    rrf_k = bm25.rrf_k if hasattr(bm25, "rrf_k") else bm25["rrf_k"]
    assert k1 == 1.5
    assert rrf_k == 60


def test_hybrid_enabled_default_flipped() -> None:
    cfg = load_config()
    assert cfg.search.hybrid_enabled is True, "M2 default flip: hybrid retrieval must default ON"
