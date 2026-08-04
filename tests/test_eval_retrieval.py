from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from kuaa.eval.datasets import EvaluationDataset, QueryCase
from kuaa.eval.retrieval import run_retrieval_eval


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


def _seed_hybrid_film(tmp_path: Path) -> SimpleNamespace:
    """Metadata + tiny CLIP index where scene 2's only 'book' evidence is a
    YOLO detection: absent from every caption (BM25-invisible) and the worst
    cosine match (CLIP-invisible). Only the metadata leg can surface it."""
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    (metadata_dir / "scene_descriptions.json").write_text(
        json.dumps(
            [
                {"scene_id": 1, "description": "an empty road"},
                {"scene_id": 2, "description": "a quiet room"},
                {"scene_id": 3, "description": "a wide field"},
            ]
        ),
        encoding="utf-8",
    )
    (metadata_dir / "keyframes_metadata.json").write_text(
        json.dumps([{"scene_id": s, "filepath": f"kf{s}.jpg"} for s in (1, 2, 3)]),
        encoding="utf-8",
    )
    (metadata_dir / "scene_tags.json").write_text(json.dumps({"interior": [2]}))
    (metadata_dir / "manual_annotations.json").write_text("{}")
    (metadata_dir / "visual_analysis.json").write_text(
        json.dumps(
            [
                {
                    "frame_path": "kf2.jpg",
                    "object_detection": {
                        "objects": [{"class": "book"}],
                        "class_counts": {"book": 1},
                    },
                }
            ]
        )
    )

    import numpy as np

    emb_dir = tmp_path / "embeddings"
    emb_dir.mkdir()
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    np.save(emb_dir / "keyframe_embeddings.npy", vectors)
    (emb_dir / "index_mapping.json").write_text(
        json.dumps(
            {
                "model": "stub",
                "dimension": 2,
                "total_vectors": 3,
                "keyframe_paths": [f"kf{s}.jpg" for s in (1, 2, 3)],
                "scene_ids": [1, 2, 3],
            }
        )
    )

    return SimpleNamespace(
        paths=SimpleNamespace(metadata_dir=str(metadata_dir), embeddings_dir=str(emb_dir)),
        embeddings=SimpleNamespace(
            filename="keyframe_embeddings.npy",
            mapping_filename="index_mapping.json",
            model="stub",
            pretrained="stub",
        ),
        search=SimpleNamespace(bm25=SimpleNamespace(k1=1.5, b=0.75, stopwords_lang=None)),
        hardware=SimpleNamespace(force_cpu=True),
    )


def test_hybrid_eval_measures_the_shipped_3way_fusion(tmp_path: Path, monkeypatch) -> None:
    """The ``hybrid`` retriever fuses the metadata leg by default (shipped
    behaviour); ``metadata_w=0.0`` is the 2-way ablation arm — and the run
    context records which one was measured."""
    import numpy as np

    import kuaa.models.registry as registry
    from kuaa.search.bm25 import clear_bm25_cache

    clear_bm25_cache()
    cfg = _seed_hybrid_film(tmp_path)

    class _StubEmbedder:
        def encode_text(self, q: str) -> np.ndarray:
            return np.array([1.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(registry, "get_image_embedder", lambda cfg, device: _StubEmbedder())

    dataset = EvaluationDataset(
        dataset="unit",
        version=1,
        queries=(QueryCase(id="q1", text="book", relevant_scene_ids=("2",), relevance={"2": 1.0}),),
    )

    shipped = run_retrieval_eval(cfg, dataset, config_path=None, retriever="hybrid", top_k=3)
    assert shipped.context["metadata_w"] == 0.65  # cfg has no override → default
    assert shipped.query_results[0].ranked_scene_ids[0] == "2"

    ablated = run_retrieval_eval(
        cfg, dataset, config_path=None, retriever="hybrid", top_k=3, metadata_w=0.0
    )
    assert ablated.context["metadata_w"] == 0.0
    assert ablated.query_results[0].ranked_scene_ids[0] != "2"
