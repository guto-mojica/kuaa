"""Evaluation utilities for demo and local collection quality checks."""

from cinemateca.eval.annotations import AnnotationStats, compute_annotation_stats
from cinemateca.eval.datasets import EvaluationDataset, QueryCase, load_dataset
from cinemateca.eval.metrics import QueryMetrics, RetrievalResult, summarize_results

__all__ = [
    "AnnotationStats",
    "EvaluationDataset",
    "QueryCase",
    "QueryMetrics",
    "RetrievalResult",
    "compute_annotation_stats",
    "load_dataset",
    "summarize_results",
]
