"""Pin audio-only search results on Jeca Tatu.

Detects silent regressions in the CLAP backend (transformers / torch
bumps, model-id changes) or the retrieval module
(:mod:`cinemateca.search.audio`). The snapshot lives at
``tests/fixtures/audio_search_regression.json`` and is committed; the
CLAP index files themselves stay gitignored under ``data/library/``.

The ``skipif`` guard makes this test graceful on fresh checkouts and CI
runners that don't carry the Jeca Tatu artefacts on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REGRESSION_PATH = Path("tests/fixtures/audio_search_regression.json")
LIBRARY_AUDIO = Path("data/library/jeca_tatu/audio/clap_embeddings.npy")


pytestmark = pytest.mark.skipif(
    not LIBRARY_AUDIO.exists(),
    reason="Jeca Tatu CLAP index not present; skip regression snapshot test.",
)


def test_audio_search_matches_jeca_tatu_snapshot() -> None:
    from api.deps import get_config
    from cinemateca.library.context import FilmContext
    from cinemateca.models.registry import get_audio_embedder
    from cinemateca.search.audio import load_audio_index, search_audio

    expected = json.loads(REGRESSION_PATH.read_text())
    cfg = get_config()
    ctx = FilmContext.for_film(cfg, "jeca_tatu")
    audio_dir = ctx.metadata_dir.parent / "audio"
    idx = load_audio_index(audio_dir)
    assert idx is not None, f"CLAP index missing at {audio_dir}"
    embedder = get_audio_embedder(cfg, device=None)
    for query, want in expected.items():
        got = search_audio(idx, embedder, query, top_k=len(want))
        # Compare scene_id order only; scores can drift by ~1e-4 across
        # CPU/GPU and across CLAP backend versions.
        assert [g["scene_id"] for g in got] == [w["scene_id"] for w in want], (
            f"Audio search order changed for query={query!r}: "
            f"got={[g['scene_id'] for g in got]} want={[w['scene_id'] for w in want]}"
        )
