"""Evaluation utilities for demo and local collection quality checks."""

from cinemateca.eval.annotations import AnnotationStats, compute_annotation_stats
from cinemateca.eval.datasets import EvaluationDataset, QueryCase, load_dataset
from cinemateca.eval.grades import EvalRun, Grade, GradeEntry, LoadedRun
from cinemateca.eval.metrics import QueryMetrics, RetrievalResult, summarize_results
from cinemateca.eval.retrieval import EvalError, RetrievalRun, run_retrieval_eval

__all__ = [
    # Annotations
    "AnnotationStats",
    "compute_annotation_stats",
    # Datasets
    "EvaluationDataset",
    "QueryCase",
    "load_dataset",
    # Grades
    "EvalRun",
    "Grade",
    "GradeEntry",
    "LoadedRun",
    # Metrics
    "QueryMetrics",
    "RetrievalResult",
    "summarize_results",
    # Retrieval
    "EvalError",
    "RetrievalRun",
    "run_retrieval_eval",
]
