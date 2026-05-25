"""Pin fusion search results on Jeca Tatu — silent-regression guard.

Skipif-guarded so it only runs on machines that have the real CLIP +
CLAP indices materialised under ``data/library/jeca_tatu/``. CI without
the artefacts skips the test, but any developer with the real library
flips it on automatically. Regenerate the snapshot via the
one-liner under Task 5.1 in ``docs/superpowers/plans/2026-05-24-m3-fusion-search.md``
whenever the fusion verb, weights, or default ``k_each`` changes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REGRESSION_PATH = Path("tests/fixtures/fusion_search_regression.json")
CLIP_PATH = Path("data/library/jeca_tatu/embeddings/keyframe_embeddings.npy")
CLAP_PATH = Path("data/library/jeca_tatu/audio/clap_embeddings.npy")


pytestmark = pytest.mark.skipif(
    not (CLIP_PATH.exists() and CLAP_PATH.exists()),
    reason="Jeca Tatu CLIP+CLAP indices not present; skip fusion snapshot.",
)


def test_fusion_search_matches_jeca_tatu_snapshot() -> None:
    from api.deps import get_config
    from api.services.search import dispatch_fusion_search
    from cinemateca.library import FilmContext

    expected = json.loads(REGRESSION_PATH.read_text())
    cfg = get_config()
    ctx = FilmContext.for_film(cfg, "jeca_tatu")
    for key, want in expected.items():
        query, w_str = key.split("|w=")
        w = float(w_str)
        hits, no_index = dispatch_fusion_search(
            cfg, ctx, query, top_k=len(want), visual_weight=w
        )
        assert not no_index, f"Unexpected no_index for {query!r} w={w}"
        got_ids = [h["scene_id"] for h in hits]
        want_ids = [w_["scene_id"] for w_ in want]
        assert got_ids == want_ids, (
            f"Fusion rank order changed for query={query!r} w={w}\n"
            f"  want: {want_ids}\n  got:  {got_ids}"
        )
