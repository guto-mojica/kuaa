"""Evaluation utilities for demo and local collection quality checks."""

from cinemateca.errors import EvalError
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
from cinemateca.eval.proxy import proxy_labels
from cinemateca.eval.retrieval import (
    RetrievalRun,
    run_audio_eval,
    run_fusion_eval,
    run_image_eval,
    run_retrieval_eval,
    run_rhyme_eval,
)
from cinemateca.eval.slates import ModalQuery, generate_slate, load_modal_queries

__all__ = [
    "AnnotationStats",
    "EvalError",
    "EvaluationDataset",
    "ModalQuery",
    "QueryCase",
    "QueryMetrics",
    "RetrievalResult",
    "RetrievalRun",
    "annotator_summary",
    "build_iaa",
    "compute_annotation_stats",
    "eval_root",
    "eval_run_id",
    "generate_slate",
    "grades_by_query",
    "grades_for_query",
    "grader_initials",
    "initials",
    "kappa_quality_label",
    "load_dataset",
    "load_modal_queries",
    "load_queries",
    "other_grades_for_current",
    "proxy_labels",
    "query_conflict_set",
    "run_audio_eval",
    "run_fusion_eval",
    "run_image_eval",
    "run_retrieval_eval",
    "run_rhyme_eval",
    "summarize_results",
]
