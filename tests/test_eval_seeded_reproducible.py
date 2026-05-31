"""E1: Reproducible eval — same seed ⇒ byte-identical metrics + seed in context.

Two test functions:

1. ``test_same_seed_identical_metrics`` — fully hermetic (BM25 only, tmp
   metadata dir, no model downloads).  Verifies that ``run_retrieval_eval``
   accepts a ``seed`` kwarg, calls it twice with the same value, and produces
   byte-identical metrics plus ``context["seed"] == 1234``.

2. ``test_run_eval_cli_records_seed`` — acceptance, requires the real
   ``jeca_tatu`` library on disk.  Invokes ``scripts/run_eval.main()`` with
   ``--retriever bm25 --seed 7 --film-slug jeca_tatu`` and asserts that the
   written ``summary.json`` contains ``context["seed"] == 7``.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cinemateca.eval.datasets import EvaluationDataset, QueryCase
from cinemateca.eval.retrieval import run_retrieval_eval

# ---------------------------------------------------------------------------
# Hermetic BM25 fixture helpers — mirror tests/test_eval_retrieval.py
# ---------------------------------------------------------------------------

JECA_METADATA_DIR = Path("data/library/jeca_tatu/metadata")


def _build_cfg(tmp_path: Path) -> SimpleNamespace:
    """Build a minimal cfg SimpleNamespace pointing at tmp_path metadata."""
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    (metadata_dir / "scene_descriptions.json").write_text(
        json.dumps(
            [
                {"scene_id": 1, "description": "railroad office paperwork"},
                {"scene_id": 2, "description": "orchard harvest wagon"},
                {"scene_id": 3, "description": "saloon dancers"},
            ]
        ),
        encoding="utf-8",
    )
    (metadata_dir / "keyframes_metadata.json").write_text(
        json.dumps(
            [
                {"scene_id": 1, "filepath": "kf1.jpg"},
                {"scene_id": 2, "filepath": "kf2.jpg"},
                {"scene_id": 3, "filepath": "kf3.jpg"},
            ]
        ),
        encoding="utf-8",
    )
    return SimpleNamespace(
        paths=SimpleNamespace(
            metadata_dir=str(metadata_dir),
            embeddings_dir=str(tmp_path / "missing-clip-index"),
        ),
        search=SimpleNamespace(bm25=SimpleNamespace(k1=1.5, b=0.75, stopwords_lang=None)),
    )


def _build_dataset() -> EvaluationDataset:
    return EvaluationDataset(
        dataset="unit",
        version=1,
        queries=(
            QueryCase(
                id="q1",
                text="orchard",
                relevant_scene_ids=("2",),
                relevance={"2": 1.0},
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Test 1: hermetic reproducibility
# ---------------------------------------------------------------------------


def test_same_seed_identical_metrics(tmp_path: Path) -> None:
    """Two runs with the same seed produce byte-identical metrics and record it."""
    cfg = _build_cfg(tmp_path)
    dataset = _build_dataset()

    run_a = run_retrieval_eval(cfg, dataset, config_path=None, retriever="bm25", top_k=3, seed=1234)
    run_b = run_retrieval_eval(cfg, dataset, config_path=None, retriever="bm25", top_k=3, seed=1234)

    assert run_a.metrics == run_b.metrics, "Two seeded runs produced different metrics"
    assert (
        run_a.context["seed"] == 1234
    ), f"Expected seed=1234 in context, got {run_a.context.get('seed')!r}"
    assert run_b.context["seed"] == 1234


# ---------------------------------------------------------------------------
# Test 2: CLI threads seed through to the written summary.json
# ---------------------------------------------------------------------------


@pytest.mark.acceptance
@pytest.mark.skipif(
    not JECA_METADATA_DIR.exists(),
    reason="Real jeca_tatu metadata not present; skipping acceptance test.",
)
def test_run_eval_cli_records_seed(tmp_path: Path) -> None:
    """CLI --seed 7 should appear in summary.json's context.seed."""
    import sys

    # Ensure scripts/ is importable (run_eval does its own sys.path insert but
    # we import it as a module here, so we need it on path ourselves).
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    import importlib

    run_eval = importlib.import_module("run_eval")

    output_dir = tmp_path / "eval_out"

    ret = run_eval.main(
        [
            "--retriever",
            "bm25",
            "--seed",
            "7",
            "--film-slug",
            "jeca_tatu",
            "--output-dir",
            str(output_dir),
            "--config",
            "config/default.yaml",
            "--queries",
            "data/eval/archive_demo_queries.yaml",
        ]
    )

    assert ret == 0, f"run_eval.main returned non-zero exit code: {ret}"

    summary_path = output_dir / "summary.json"
    assert summary_path.exists(), f"Expected {summary_path} to be written by run_eval"

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "context" in payload, "summary.json missing 'context' key"
    assert (
        payload["context"].get("seed") == 7
    ), f"Expected context.seed==7, got {payload['context'].get('seed')!r}"
