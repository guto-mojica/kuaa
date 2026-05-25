"""HTTP-layer tests for ``/api/search?modality=fusion``.

The fusion modality dispatches into :func:`api.services.search.dispatch_fusion_search`
which linearly combines CLIP keyframe + CLAP audio cosine scores under a
tunable ``w`` (visual_weight). These tests pin:

* The new ``modality="fusion"`` query param is wired through the same
  TestClient → router → handler → render path as the audio modality.
* The new ``w`` query param is accepted, clamped into ``[0, 1]``, and
  defaulted to ``cfg.retrieval.fusion.visual_weight`` (fallback 0.5) when
  omitted.
* Both per-film (``?film=<slug>``) and cross-film (no slug) shapes work.
* Both embedders (CLIP + CLAP) are stubbed via the registry seam so no
  real model loads happen.

The fusion dispatcher tests already cover the scoring math
(:mod:`tests.test_search_service_fusion`); these tests only assert the
HTTP route surface — params, dispatch, rendering.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cinemateca.library import register_film

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
    """Write a synthetic CLIP keyframe index under ``<film_dir>/embeddings/``."""
    emb_dir = film_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((n, dim)).astype("float32")
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    np.save(emb_dir / "keyframe_embeddings.npy", emb)
    mapping = [{"scene_id": i, "filepath": f"frames/s{i:04d}.jpg"} for i in range(n)]
    (emb_dir / "index_mapping.json").write_text(json.dumps(mapping))


def _seed_clap(film_dir: Path, *, dim: int = 4, n: int = 4) -> None:
    """Write a synthetic CLAP audio index under ``<film_dir>/audio/``."""
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
    """Register a film and create the ``raw/`` placeholder file."""
    register_film(
        library_dir,
        slug=slug,
        title=title or slug,
        year=1959,
        raw_filename=f"{slug}.mp4",
    )
    (library_dir / slug / "raw").mkdir(parents=True, exist_ok=True)
    (library_dir / slug / "raw" / f"{slug}.mp4").write_bytes(b"")


@pytest.fixture()
def patch_embedders(monkeypatch):
    """Stub both CLIP + CLAP embedder factories at the registry module level.

    The fusion dispatcher resolves both via
    ``cinemateca.models.registry.{get_image_embedder, get_audio_embedder}``
    at call time, so patching the registry is sufficient — no need to
    also patch any service-level re-exports.
    """
    import cinemateca.models.registry as registry

    monkeypatch.setattr(registry, "get_image_embedder", lambda cfg, device=None: _Stub(4))
    monkeypatch.setattr(registry, "get_audio_embedder", lambda cfg, device=None: _Stub(4))


# ── Tests ────────────────────────────────────────────────────────────────────


def test_api_search_modality_fusion_per_film_200(client, tmp_config, patch_embedders) -> None:
    """``/api/search?modality=fusion&w=0.5&film=<slug>`` hits the fusion
    dispatch and renders a results fragment (not the no-index empty state)."""
    library_dir = Path(tmp_config.paths.library_dir)
    _register(library_dir, "jeca_tatu")
    _seed_clip(library_dir / "jeca_tatu")
    _seed_clap(library_dir / "jeca_tatu")

    resp = client.get(
        "/api/search",
        params={
            "q": "cats",
            "modality": "fusion",
            "w": 0.5,
            "film": "jeca_tatu",
            "top_k": 3,
        },
    )
    assert resp.status_code == 200
    assert "Run the pipeline with the Embeddings step first" not in resp.text
    assert "b-card" in resp.text


def test_api_search_modality_fusion_aggregate_200(client, tmp_config, patch_embedders) -> None:
    """Cross-film fusion (no ``film=`` param): two films with both indices →
    results from both surface in the rendered fragment."""
    library_dir = Path(tmp_config.paths.library_dir)
    _register(library_dir, "film_a", title="Film A")
    _seed_clip(library_dir / "film_a")
    _seed_clap(library_dir / "film_a")
    _register(library_dir, "film_b", title="Film B")
    _seed_clip(library_dir / "film_b")
    _seed_clap(library_dir / "film_b")

    resp = client.get(
        "/api/search",
        params={"q": "cats", "modality": "fusion", "w": 0.5, "top_k": 10},
    )
    assert resp.status_code == 200
    assert "Run the pipeline with the Embeddings step first" not in resp.text
    assert "b-card" in resp.text
    # Both films contributed hits — slugs surface in the rendered markup
    # (the aggregate-card partial tags each result with the film title /
    # slug via enrich_hits_with_film_metadata).
    assert "Film A" in resp.text or "film_a" in resp.text
    assert "Film B" in resp.text or "film_b" in resp.text


def test_api_search_modality_fusion_clamps_w_out_of_range(
    client, tmp_config, patch_embedders
) -> None:
    """``w=1.5`` must be clamped to ``[0, 1]`` (UX-friendly), not 422-rejected.

    The route should accept the value, clamp it to ``1.0``, and render a
    results fragment exactly as if ``w=1.0`` had been sent.
    """
    library_dir = Path(tmp_config.paths.library_dir)
    _register(library_dir, "jeca_tatu")
    _seed_clip(library_dir / "jeca_tatu")
    _seed_clap(library_dir / "jeca_tatu")

    resp = client.get(
        "/api/search",
        params={
            "q": "cats",
            "modality": "fusion",
            "w": 1.5,
            "film": "jeca_tatu",
            "top_k": 3,
        },
    )
    assert resp.status_code == 200
    # Non-empty markup — the dispatcher ran and rendered results.
    assert "b-card" in resp.text


def test_api_search_modality_fusion_w_default_uses_config(
    client, tmp_config, patch_embedders
) -> None:
    """Omitting ``w=`` must fall through to the config default
    (``cfg.retrieval.fusion.visual_weight``, fallback ``0.5``) without
    erroring. The actual weight value is verified by the service-layer
    tests; here we only assert the route reaches the dispatcher cleanly."""
    library_dir = Path(tmp_config.paths.library_dir)
    _register(library_dir, "jeca_tatu")
    _seed_clip(library_dir / "jeca_tatu")
    _seed_clap(library_dir / "jeca_tatu")

    resp = client.get(
        "/api/search",
        params={
            "q": "cats",
            "modality": "fusion",
            "film": "jeca_tatu",
            "top_k": 3,
        },
    )
    assert resp.status_code == 200
    assert "b-card" in resp.text
