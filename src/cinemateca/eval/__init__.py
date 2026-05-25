"""Evaluation utilities for demo and local collection quality checks."""

from cinemateca.eval.annotations import AnnotationStats, compute_annotation_stats
from cinemateca.eval.datasets import EvaluationDataset, QueryCase, load_dataset, load_queries
from cinemateca.eval.grades import grades_by_query, grades_for_query
from cinemateca.eval.metrics import QueryMetrics, RetrievalResult, summarize_results
from cinemateca.eval.paths import eval_root, eval_run_id

__all__ = [
    "AnnotationStats",
    "EvaluationDataset",
    "QueryCase",
    "QueryMetrics",
    "RetrievalResult",
    "compute_annotation_stats",
    "eval_root",
    "eval_run_id",
    "grades_by_query",
    "grades_for_query",
    "load_dataset",
    "load_queries",
    "summarize_results",
]
