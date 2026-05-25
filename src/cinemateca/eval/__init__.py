"""Evaluation utilities for demo and local collection quality checks."""

from cinemateca.eval.annotations import AnnotationStats, compute_annotation_stats
from cinemateca.eval.datasets import EvaluationDataset, QueryCase, load_dataset, load_queries
from cinemateca.eval.grader_metrics import (
    annotator_summary,
    build_iaa,
    grader_initials,
    initials,
    kappa_quality_label,
    other_grades_for_current,
    query_conflict_set,
)
from cinemateca.eval.grades import grades_by_query, grades_for_query
from cinemateca.eval.metrics import QueryMetrics, RetrievalResult, summarize_results
from cinemateca.eval.paths import eval_root, eval_run_id

__all__ = [
    "AnnotationStats",
    "EvaluationDataset",
    "QueryCase",
    "QueryMetrics",
    "RetrievalResult",
    "annotator_summary",
    "build_iaa",
    "compute_annotation_stats",
    "eval_root",
    "eval_run_id",
    "grades_by_query",
    "grades_for_query",
    "grader_initials",
    "initials",
    "kappa_quality_label",
    "load_dataset",
    "load_queries",
    "other_grades_for_current",
    "query_conflict_set",
    "summarize_results",
]
