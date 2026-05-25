"""Tests for ``api.services.search.dispatch_fusion_search``.

This is the M3 fusion entry point that mirrors :func:`dispatch_audio_search`:

* ``ctx=None``  → cross-film aggregate over the registry.
* ``ctx`` given → per-film fusion.

Both paths return ``(hits, no_index)``. ``no_index=True`` when neither
CLIP nor CLAP indices exist for any candidate film. Embedders are stubbed
via ``cinemateca.models.registry.{get_image_embedder, get_audio_embedder}``
so no real model loads happen in these tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cinemateca.library import FilmContext, register_film

# ── Stub embedders (CLIP + CLAP share the encode_text shape) ─────────────────


class _Stub:
    """Stub embedder. ``encode_text`` returns a fixed unit vector of ``dim``."""

    def __init__(self, dim: int) -> None:
        self._dim = dim

    def encode_text(self, text: str) -> np.ndarray:
        v = np.ones(self._dim, dtype="float32")
        return v / np.linalg.norm(v)


# ── Fixture helpers ──────────────────────────────────────────────────────────


def _seed_clip(film_dir: Path, *, dim: int = 4, n: int = 4) -> None:
    """Write a synthetic CLIP keyframe index under ``<film_dir>/embeddings/``.

    Rows are basis-like vectors so cosines vs the ``_Stub(dim)`` unit-query
    are deterministic and not all-zero. The exact ordering doesn't matter
    for fusion tests — we only assert presence, slug tagging, and that
    monotonic-descending property of the final sort.
    """
    emb_dir = film_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((n, dim)).astype("float32")
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    np.save(emb_dir / "keyframe_embeddings.npy", emb)
    mapping = [{"scene_id": i, "filepath": f"frames/s{i:04d}.jpg"} for i in range(n)]
    (emb_dir / "index_mapping.json").write_text(json.dumps(mapping))


def _seed_clap(film_dir: Path, *, dim: int = 4, n: int = 4) -> None:
    """Write a synthetic CLAP audio index under ``<film_dir>/audio/``.

    Uses the ``dict-of-parallel-arrays`` shape that ``ClapHFEmbedder.save``
    produces in production — the loader normalises both this and the
    row-shaped list, so either is fine.
    """
    audio_dir = film_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    emb = rng.standard_normal((n, dim)).astype("float32")
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    np.save(audio_dir / "clap_embeddings.npy", emb)
    (audio_dir / "audio_mapping.json").write_text(
        json.dumps(
            {
                "model": "stub",
                "dimension": dim,
                "total_vectors": n,
                "normalized": True,
                "scene_ids": list(range(n)),
                "wav_paths": [f"audio/scene_{i:04d}.wav" for i in range(n)],
                "start_times_s": [float(i * 5) for i in range(n)],
                "end_times_s": [float((i + 1) * 5) for i in range(n)],
            }
        )
    )


def _register(library_dir: Path, slug: str, *, title: str = "") -> None:
    """Register a film in ``library_dir`` and create the ``raw/`` placeholder."""
    register_film(
        library_dir,
        slug=slug,
        title=title or slug,
        year=1959,
        raw_filename=f"{slug}.mp4",
    )
    (library_dir / slug / "raw").mkdir(parents=True, exist_ok=True)
    (library_dir / slug / "raw" / f"{slug}.mp4").write_bytes(b"")


def _patch_embedders(monkeypatch, *, clip_dim: int = 4, clap_dim: int = 4) -> None:
    """Stub both embedders at the registry module level."""
    import cinemateca.models.registry as registry

    monkeypatch.setattr(registry, "get_image_embedder", lambda cfg, device=None: _Stub(clip_dim))
    monkeypatch.setattr(registry, "get_audio_embedder", lambda cfg, device=None: _Stub(clap_dim))


# ── Tests ────────────────────────────────────────────────────────────────────


def test_dispatch_fusion_search_per_film_returns_hits(tmp_config, monkeypatch) -> None:
    """Per-film fusion: CLIP + CLAP both present → ranked hits with both scores."""
    from api.services.search import dispatch_fusion_search

    library_dir = Path(tmp_config.paths.library_dir)
    _register(library_dir, "jeca_tatu")
    _seed_clip(library_dir / "jeca_tatu")
    _seed_clap(library_dir / "jeca_tatu")
    _patch_embedders(monkeypatch)

    ctx = FilmContext.for_film(tmp_config, "jeca_tatu")
    hits, no_index = dispatch_fusion_search(tmp_config, ctx, "x", top_k=5)

    assert no_index is False
    assert 0 < len(hits) <= 5
    for h in hits:
        assert "clip_score" in h
        assert "clap_score" in h
        assert h["film_slug"] == "jeca_tatu"
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)
    # Default visual_weight=0.5 → score == 0.5*clip + 0.5*clap. Asserting the
    # linear-fusion formula here guards against silent reweighting.
    for h in hits:
        assert h["score"] == pytest.approx(
            0.5 * h["clip_score"] + 0.5 * h["clap_score"], abs=1e-5
        ), f"Fusion formula violated for {h}"


def test_dispatch_fusion_search_per_film_missing_audio_falls_back_to_clip_only(
    tmp_config, monkeypatch
) -> None:
    """CLIP present, CLAP missing → still returns hits, ``clap_score == 0.0``."""
    from api.services.search import dispatch_fusion_search

    library_dir = Path(tmp_config.paths.library_dir)
    _register(library_dir, "jeca_tatu")
    _seed_clip(library_dir / "jeca_tatu")
    _patch_embedders(monkeypatch)

    ctx = FilmContext.for_film(tmp_config, "jeca_tatu")
    hits, no_index = dispatch_fusion_search(tmp_config, ctx, "x", top_k=5)

    assert no_index is False
    assert len(hits) > 0
    for h in hits:
        assert h["clap_score"] == pytest.approx(0.0)
        assert h["film_slug"] == "jeca_tatu"


def test_dispatch_fusion_search_per_film_missing_both_returns_no_index(
    tmp_config, monkeypatch
) -> None:
    """Neither modality present → ``([], True)``, no embedder load."""
    from api.services.search import dispatch_fusion_search

    library_dir = Path(tmp_config.paths.library_dir)
    _register(library_dir, "jeca_tatu")
    _patch_embedders(monkeypatch)

    ctx = FilmContext.for_film(tmp_config, "jeca_tatu")
    hits, no_index = dispatch_fusion_search(tmp_config, ctx, "x", top_k=5)
    assert hits == []
    assert no_index is True


def test_dispatch_fusion_search_aggregate_returns_hits_from_both_films(
    tmp_config, monkeypatch
) -> None:
    """Cross-film: two films, both with CLIP + CLAP → both slugs appear in hits."""
    from api.services.search import dispatch_fusion_search

    library_dir = Path(tmp_config.paths.library_dir)
    _register(library_dir, "film_a", title="Film A")
    _seed_clip(library_dir / "film_a")
    _seed_clap(library_dir / "film_a")
    _register(library_dir, "film_b", title="Film B")
    _seed_clip(library_dir / "film_b")
    _seed_clap(library_dir / "film_b")
    _patch_embedders(monkeypatch)

    hits, no_index = dispatch_fusion_search(tmp_config, None, "x", top_k=10)

    assert no_index is False
    slugs = {h["film_slug"] for h in hits}
    assert slugs == {"film_a", "film_b"}
    for h in hits:
        # Cross-film hits carry the title too, per the audio dispatcher contract.
        assert h["film_title"] in {"Film A", "Film B"}
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_dispatch_fusion_search_aggregate_skips_films_without_indices(
    tmp_config, monkeypatch
) -> None:
    """3 films — only the first + third have indices → middle is skipped silently."""
    from api.services.search import dispatch_fusion_search

    library_dir = Path(tmp_config.paths.library_dir)
    _register(library_dir, "film_a")
    _seed_clip(library_dir / "film_a")
    _seed_clap(library_dir / "film_a")
    _register(library_dir, "film_b")  # NO indices
    _register(library_dir, "film_c")
    _seed_clip(library_dir / "film_c")
    _seed_clap(library_dir / "film_c")
    _patch_embedders(monkeypatch)

    hits, no_index = dispatch_fusion_search(tmp_config, None, "x", top_k=10)

    assert no_index is False
    slugs = {h["film_slug"] for h in hits}
    assert "film_b" not in slugs
    assert slugs.issubset({"film_a", "film_c"})
    # At least one of the seeded films must contribute results.
    assert slugs


def test_dispatch_fusion_search_aggregate_no_films_returns_no_index(
    tmp_config, monkeypatch
) -> None:
    """Empty registry → ``([], True)``."""
    from api.services.search import dispatch_fusion_search

    _patch_embedders(monkeypatch)
    hits, no_index = dispatch_fusion_search(tmp_config, None, "x", top_k=5)
    assert hits == []
    assert no_index is True


def test_dispatch_fusion_search_aggregate_loads_each_embedder_once(tmp_config, monkeypatch) -> None:
    """Aggregate path must instantiate get_image_embedder + get_audio_embedder
    at most once across all films, not once per film. Mirrors the
    dispatch_audio_search contract."""
    from api.services.search import dispatch_fusion_search

    library_dir = Path(tmp_config.paths.library_dir)
    _register(library_dir, "film_a", title="Film A")
    _seed_clip(library_dir / "film_a")
    _seed_clap(library_dir / "film_a")
    _register(library_dir, "film_b", title="Film B")
    _seed_clip(library_dir / "film_b")
    _seed_clap(library_dir / "film_b")

    call_counts = {"image": 0, "audio": 0}

    def _img(cfg, device=None):
        call_counts["image"] += 1
        return _Stub(4)

    def _aud(cfg, device=None):
        call_counts["audio"] += 1
        return _Stub(4)

    import cinemateca.models.registry as registry

    monkeypatch.setattr(registry, "get_image_embedder", _img)
    monkeypatch.setattr(registry, "get_audio_embedder", _aud)

    hits, no_index = dispatch_fusion_search(tmp_config, None, "x", top_k=5)

    assert no_index is False
    assert len(hits) > 0
    assert call_counts == {
        "image": 1,
        "audio": 1,
    }, f"Expected each embedder loaded exactly once across films, got {call_counts}"
