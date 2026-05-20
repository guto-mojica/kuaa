"""Retrieval metric math for evaluation reports."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from cinemateca.scene_ids import scene_id_key


@dataclass(frozen=True)
class QueryMetrics:
    """Metric values for one query."""

    recall_at_5: float
    recall_at_10: float
    reciprocal_rank: float
    ndcg_at_10: float


@dataclass(frozen=True)
class RetrievalResult:
    """Ranked retrieval output and metrics for one query."""

    query_id: str
    text: str
    relevant_scene_ids: tuple[str, ...]
    ranked_scene_ids: tuple[str, ...]
    metrics: QueryMetrics
    top_results: tuple[dict[str, Any], ...] = ()
    missing_relevant_scene_ids: tuple[str, ...] = ()


def _canonical_ids(values) -> tuple[str, ...]:
    return tuple(scene_id_key(v) for v in values)


def recall_at_k(ranked_scene_ids, relevant_scene_ids, k: int) -> float:
    """Mean-recall component for one query."""

    relevant = set(_canonical_ids(relevant_scene_ids))
    if not relevant:
        raise ValueError("relevant_scene_ids must not be empty")
    retrieved = set(_canonical_ids(ranked_scene_ids)[:k])
    return len(retrieved & relevant) / len(relevant)


def reciprocal_rank(ranked_scene_ids, relevant_scene_ids) -> float:
    """Return reciprocal rank of the first relevant result, or 0."""

    relevant = set(_canonical_ids(relevant_scene_ids))
    if not relevant:
        raise ValueError("relevant_scene_ids must not be empty")
    for idx, sid in enumerate(_canonical_ids(ranked_scene_ids), start=1):
        if sid in relevant:
            return 1.0 / idx
    return 0.0


def dcg_at_k(ranked_scene_ids, relevance: dict[str, float], k: int) -> float:
    """Discounted cumulative gain for ranked scene ids."""

    grades = {scene_id_key(k): float(v) for k, v in relevance.items()}
    score = 0.0
    for idx, sid in enumerate(_canonical_ids(ranked_scene_ids)[:k], start=1):
        grade = grades.get(sid, 0.0)
        if grade <= 0:
            continue
        score += (2.0**grade - 1.0) / math.log2(idx + 1)
    return score


def ndcg_at_k(ranked_scene_ids, relevance: dict[str, float], k: int) -> float:
    """Normalized DCG for graded relevance labels."""

    positive_grades = sorted(
        (float(v) for v in relevance.values() if float(v) > 0),
        reverse=True,
    )
    if not positive_grades:
        raise ValueError("relevance must contain at least one positive grade")

    ideal_ids = tuple(str(i) for i in range(len(positive_grades)))
    ideal_relevance = {sid: grade for sid, grade in zip(ideal_ids, positive_grades)}
    ideal = dcg_at_k(ideal_ids, ideal_relevance, k)
    if ideal == 0:
        return 0.0
    return dcg_at_k(ranked_scene_ids, relevance, k) / ideal


def evaluate_query(
    *,
    query_id: str,
    text: str,
    relevant_scene_ids,
    ranked_scene_ids,
    relevance: dict[str, float],
    top_results: tuple[dict[str, Any], ...] = (),
    index_scene_ids=None,
) -> RetrievalResult:
    """Compute retrieval metrics for one query."""

    relevant = _canonical_ids(relevant_scene_ids)
    ranked = _canonical_ids(ranked_scene_ids)
    if not relevance:
        relevance = {sid: 1.0 for sid in relevant}
    metrics = QueryMetrics(
        recall_at_5=recall_at_k(ranked, relevant, 5),
        recall_at_10=recall_at_k(ranked, relevant, 10),
        reciprocal_rank=reciprocal_rank(ranked, relevant),
        ndcg_at_10=ndcg_at_k(ranked, relevance, 10),
    )

    missing: tuple[str, ...] = ()
    if index_scene_ids is not None:
        available = set(_canonical_ids(index_scene_ids))
        missing = tuple(sid for sid in relevant if sid not in available)

    return RetrievalResult(
        query_id=query_id,
        text=text,
        relevant_scene_ids=relevant,
        ranked_scene_ids=ranked,
        metrics=metrics,
        top_results=top_results,
        missing_relevant_scene_ids=missing,
    )


def summarize_results(results: list[RetrievalResult]) -> dict[str, float | int]:
    """Average query metrics across a retrieval run."""

    if not results:
        raise ValueError("results must not be empty")

    n = len(results)
    return {
        "query_count": n,
        "recall_at_5": sum(r.metrics.recall_at_5 for r in results) / n,
        "recall_at_10": sum(r.metrics.recall_at_10 for r in results) / n,
        "mrr": sum(r.metrics.reciprocal_rank for r in results) / n,
        "ndcg_at_10": sum(r.metrics.ndcg_at_10 for r in results) / n,
    }
