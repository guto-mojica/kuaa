from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from cinemateca.eval.datasets import EvaluationDataset, QueryCase
from cinemateca.eval.retrieval import run_retrieval_eval


def test_bm25_retrieval_eval_does_not_require_clip_index(tmp_path: Path) -> None:
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

    cfg = SimpleNamespace(
        paths=SimpleNamespace(
            metadata_dir=str(metadata_dir),
            embeddings_dir=str(tmp_path / "missing-clip-index"),
        ),
        search=SimpleNamespace(bm25=SimpleNamespace(k1=1.5, b=0.75, stopwords_lang=None)),
    )
    dataset = EvaluationDataset(
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

    run = run_retrieval_eval(
        cfg,
        dataset,
        config_path=None,
        retriever="bm25",
        top_k=3,
    )

    assert run.context["retriever"] == "bm25"
    assert run.context["embeddings_path"] == ""
    assert run.context["model"] == "BM25"
    assert run.query_results[0].ranked_scene_ids[0] == "2"
    assert run.query_results[0].top_results[0]["filepath"] == "kf2.jpg"
