"""HTTP-layer tests for ``/api/search?modality=audio``.

The audio modality dispatches into ``cinemateca.search.audio`` (the CLAP
joint text+audio space) instead of CLIP. These tests pin:

* The new ``modality`` query param is wired and ``"audio"`` reaches the
  audio dispatch handler (TestClient → router → handler → render).
* A film whose CLAP index is missing renders the no-index empty state
  rather than 500-ing.
* The CLAP backend itself is NOT loaded — ``get_audio_embedder`` is
  monkeypatched to a tiny in-process stub so the test runs without the
  multi-second HuggingFace model download.

The fixture uses the REAL ``ClapHFEmbedder.save()`` dict-of-parallel-
arrays mapping shape; the loader (``cinemateca.search.audio``) normalises
both shapes, so this exercises the production path.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cinemateca.library import register_film

SLUG = "audio_film"


def _seed_clap_film(cfg) -> None:
    """Register one film and write a synthetic CLAP index for it.

    ``embeddings[0]`` is unit-vector along axis 0 so it scores ``1.0``
    against the stub query vector (also axis-0 unit) — the rest of the
    rows are random noise and score near 0. The top-1 ordering is
    therefore deterministic.
    """
    library_dir = Path(cfg.paths.library_dir)
    register_film(
        library_dir,
        slug=SLUG,
        title="Audio Film",
        year=1959,
        raw_filename=f"{SLUG}.mp4",
    )
    film_dir = library_dir / SLUG
    (film_dir / "raw").mkdir(parents=True, exist_ok=True)
    (film_dir / "raw" / f"{SLUG}.mp4").write_bytes(b"")
    audio_dir = film_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    emb = rng.standard_normal((6, 512)).astype("float32")
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    # Plant a deterministic top-1: row 0 is the axis-0 unit vector, so
    # cosine vs the stub query (also axis-0 unit) is exactly 1.0.
    emb[0] = 0.0
    emb[0, 0] = 1.0
    np.save(audio_dir / "clap_embeddings.npy", emb)

    # Real ClapHFEmbedder.save() shape: dict-of-parallel-arrays. The
    # loader normalises this to the row-shaped AudioMappingRow list.
    (audio_dir / "audio_mapping.json").write_text(
        json.dumps(
            {
                "model": "laion/larger_clap_general",
                "dimension": 512,
                "total_vectors": 6,
                "normalized": True,
                "scene_ids": list(range(6)),
                "wav_paths": [f"audio/scene_{i:04d}.wav" for i in range(6)],
                "start_times_s": [float(i * 5) for i in range(6)],
                "end_times_s": [float((i + 1) * 5) for i in range(6)],
            }
        )
    )


class _StubAudioEmbedder:
    """Returns a deterministic axis-0 unit query vector. No CLAP load."""

    def encode_text(self, text: str) -> np.ndarray:  # noqa: D401 — Protocol shape
        v = np.zeros(512, dtype="float32")
        v[0] = 1.0
        return v


@pytest.fixture()
def patch_audio_embedder(monkeypatch):
    """Stub the audio embedder factory at the module the route imports it from.

    The route does ``from cinemateca.models.registry import get_audio_embedder``
    at call time (inside the audio dispatch handler), so patching the
    registry's binding is sufficient — no need to also patch a route-level
    re-export.
    """
    import cinemateca.models.registry as registry

    monkeypatch.setattr(
        registry, "get_audio_embedder", lambda cfg, device=None: _StubAudioEmbedder()
    )


# ── Tests ────────────────────────────────────────────────────────────────────


def test_api_search_modality_audio_per_film_returns_results(
    client, tmp_config, patch_audio_embedder
) -> None:
    """``/api/search?modality=audio&film=<slug>`` hits the CLAP dispatch
    and renders a results fragment (not the no-index empty state)."""
    _seed_clap_film(tmp_config)
    resp = client.get(
        "/api/search",
        params={
            "q": "festive music",
            "modality": "audio",
            "film": SLUG,
            "top_k": 3,
        },
    )
    assert resp.status_code == 200
    # Results fragment renders ``.b-card`` markers, never the no-index
    # message. The no-index path emits the literal English message
    # (locale pinned to ``en`` by the ``client`` fixture).
    assert "Run the pipeline with the Embeddings step first" not in resp.text
    assert "b-card" in resp.text


def test_api_search_modality_audio_no_index_renders_empty_state(
    client, tmp_config, patch_audio_embedder
) -> None:
    """A film with no CLAP index renders the no-index empty state
    instead of 500-ing."""
    library_dir = Path(tmp_config.paths.library_dir)
    register_film(
        library_dir,
        slug="blank",
        title="Blank",
        year=2026,
        raw_filename="blank.mp4",
    )
    (library_dir / "blank" / "raw").mkdir(parents=True, exist_ok=True)
    (library_dir / "blank" / "raw" / "blank.mp4").write_bytes(b"")
    resp = client.get(
        "/api/search",
        params={"q": "anything", "modality": "audio", "film": "blank"},
    )
    assert resp.status_code == 200
    # The no-index empty-state is the same one CLIP renders.
    assert "Run the pipeline with the Embeddings step first" in resp.text


def test_api_search_modality_audio_aggregate_walks_films(
    client, tmp_config, patch_audio_embedder
) -> None:
    """Cross-film audio search (no ``film=`` param) walks every registered
    film with a CLAP index and merges top-k by score. Films without a
    CLAP index are silently skipped."""
    _seed_clap_film(tmp_config)
    # A second film registered but WITHOUT a CLAP index — must be skipped
    # cleanly (not raise) so the seeded film's hits still surface.
    library_dir = Path(tmp_config.paths.library_dir)
    register_film(
        library_dir,
        slug="silent",
        title="Silent",
        year=1920,
        raw_filename="silent.mp4",
    )
    (library_dir / "silent" / "raw").mkdir(parents=True, exist_ok=True)
    (library_dir / "silent" / "raw" / "silent.mp4").write_bytes(b"")

    resp = client.get(
        "/api/search",
        params={"q": "festive music", "modality": "audio", "top_k": 3},
    )
    assert resp.status_code == 200
    assert "Run the pipeline with the Embeddings step first" not in resp.text
    assert "b-card" in resp.text


def test_api_search_default_modality_is_text_preserved(client) -> None:
    """Omitting ``modality`` must not regress the text path: the route
    falls through to the existing CLIP/BM25/hybrid dispatch unchanged."""
    # Empty library → no-index. The point of this test is that the route
    # accepts a request without ``modality`` and reaches the text branch
    # (which then 200s with the no-index fragment). A regression that
    # broke the default would 500 or 422.
    resp = client.get("/api/search", params={"q": "menina"})
    assert resp.status_code == 200


def test_api_search_modality_audio_short_query_returns_empty(
    client, tmp_config, patch_audio_embedder
) -> None:
    """A < 2-char query short-circuits with an empty body — same contract
    as the text path. Prevents the dispatcher from loading the audio
    index for keystroke noise."""
    _seed_clap_film(tmp_config)
    resp = client.get(
        "/api/search",
        params={"q": "x", "modality": "audio", "film": SLUG},
    )
    assert resp.status_code == 200
    assert resp.text == ""
