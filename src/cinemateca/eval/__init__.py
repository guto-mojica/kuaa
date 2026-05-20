"""Evaluation utilities for demo and local collection quality checks."""

from cinemateca.eval.datasets import EvaluationDataset, QueryCase, load_dataset
from cinemateca.eval.metrics import QueryMetrics, RetrievalResult, summarize_results

__all__ = [
    "EvaluationDataset",
    "QueryCase",
    "QueryMetrics",
    "RetrievalResult",
    "load_dataset",
    "summarize_results",
]
