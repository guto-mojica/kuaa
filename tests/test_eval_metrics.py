from __future__ import annotations

from pathlib import Path

import pytest

from cinemateca.eval.datasets import DatasetError, load_dataset
from cinemateca.eval.metrics import (
    evaluate_query,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    summarize_results,
)


@pytest.mark.skipif(
    not Path("data/eval/archive_demo_queries.yaml").exists(),
    reason="Archive demo queries file not available",
)
def test_archive_demo_query_file_loads():
    dataset = load_dataset("data/eval/archive_demo_queries.yaml")

    assert dataset.dataset == "archive_demo"
    assert dataset.version == 1
    assert len(dataset.queries) >= 20
    assert dataset.queries[0].id == "q001"
    assert dataset.queries[0].relevant_scene_ids == ("1", "9")


def test_dataset_rejects_query_without_relevant_scene(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
dataset: bad
version: 1
queries:
  - id: q001
    text: no labels
    relevant_scene_ids: []
""",
        encoding="utf-8",
    )

    with pytest.raises(DatasetError, match="relevant_scene_ids"):
        load_dataset(path)


def test_recall_and_reciprocal_rank_normalize_scene_ids():
    ranked = ["x", 2.0, "1", "3"]
    relevant = [1, "2"]

    assert recall_at_k(ranked, relevant, 1) == 0.0
    assert recall_at_k(ranked, relevant, 3) == 1.0
    assert reciprocal_rank(ranked, relevant) == pytest.approx(0.5)


def test_ndcg_uses_graded_relevance():
    ranked = ["x", "b", "a"]
    relevance = {"a": 3, "b": 1}

    assert ndcg_at_k(ranked, relevance, 10) == pytest.approx(0.541, abs=0.001)


def test_evaluate_query_and_summary():
    a = evaluate_query(
        query_id="q001",
        text="alpha",
        relevant_scene_ids=[1, 2],
        ranked_scene_ids=[9, 2, 1],
        relevance={"1": 3, "2": 1},
        index_scene_ids=[1, 2, 9],
    )
    b = evaluate_query(
        query_id="q002",
        text="beta",
        relevant_scene_ids=[3],
        ranked_scene_ids=[9, 8],
        relevance={"3": 1},
        index_scene_ids=[8, 9],
    )

    assert a.metrics.recall_at_5 == 1.0
    assert a.metrics.reciprocal_rank == pytest.approx(0.5)
    assert b.metrics.recall_at_10 == 0.0
    assert b.missing_relevant_scene_ids == ("3",)

    summary = summarize_results([a, b])
    assert summary["query_count"] == 2
    assert summary["recall_at_5"] == pytest.approx(0.5)
    assert summary["mrr"] == pytest.approx(0.25)
