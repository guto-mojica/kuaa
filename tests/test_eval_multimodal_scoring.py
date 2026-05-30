"""GATE (E3b): run_eval.py scores all five modality types on the demo corpus.

This is the literal E3 *Done when*: "run_eval.py scores all five modality
types on the demo corpus." For each modality in
``("text", "image", "audio", "fusion", "rhyme")`` it invokes
``scripts/run_eval.main(...)`` against the real ``data/eval/m3_full_queries.yaml``
and the on-disk ``jeca_tatu`` index, then asserts the per-modality
``summary.json`` exists and carries a metrics block with ``query_count >= 1``
and the four summary keys.

This is the HEAVY run: text/image use CLIP, audio/fusion use CLAP (both
GPU-accelerated on the local box). It is ``@pytest.mark.acceptance`` and
skips cleanly when the jeca_tatu CLIP embeddings are absent.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_JECA_CLIP = _REPO_ROOT / "data/library/jeca_tatu/embeddings/keyframe_embeddings.npy"
_M3_QUERIES = _REPO_ROOT / "data/eval/m3_full_queries.yaml"

_MODALITIES = ("text", "image", "audio", "fusion", "rhyme")
_METRIC_KEYS = ("recall_at_5", "recall_at_10", "mrr", "ndcg_at_10")


@pytest.mark.acceptance
@pytest.mark.skipif(
    not _JECA_CLIP.exists(),
    reason="jeca_tatu CLIP index not present; skipping multimodal scoring GATE.",
)
def test_run_eval_scores_each_modality_on_demo_corpus(tmp_path: Path) -> None:
    """Every modality writes a summary.json with >=1 scored query + 4 metric keys."""
    scripts_dir = _REPO_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    run_eval = importlib.import_module("run_eval")

    for modality in _MODALITIES:
        out_dir = tmp_path / modality
        ret = run_eval.main(
            [
                "--config",
                "config/default.yaml",
                "--queries",
                str(_M3_QUERIES),
                "--modality",
                modality,
                "--film-slug",
                "jeca_tatu",
                "--output-dir",
                str(out_dir),
                "--seed",
                "0",
            ]
        )
        assert ret == 0, f"[{modality}] run_eval.main returned {ret}"

        summary_path = out_dir / "summary.json"
        assert summary_path.exists(), f"[{modality}] missing {summary_path}"

        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics", {})
        assert metrics.get("query_count", 0) >= 1, (
            f"[{modality}] query_count must be >= 1, got {metrics.get('query_count')!r}"
        )
        for key in _METRIC_KEYS:
            assert key in metrics, f"[{modality}] summary.json metrics missing {key!r}"
