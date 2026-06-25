"""Existing exceptions must inherit the F2 taxonomy (back-compat preserved)."""

from __future__ import annotations

from kuaa.errors import (
    ArtefactError,
    RetrievalError,
    UserInputError,
)


def test_domain_error_is_user_input():
    from kuaa.domain import DomainError

    assert issubclass(DomainError, UserInputError)


def test_dataset_error_is_user_input():
    from kuaa.eval.datasets import DatasetError

    assert issubclass(DatasetError, UserInputError)


def test_eval_error_is_retrieval():
    from kuaa.eval.retrieval import EvalError

    assert issubclass(EvalError, RetrievalError)


def test_annotation_stats_error_is_artefact():
    from kuaa.eval.annotations import AnnotationStatsError

    assert issubclass(AnnotationStatsError, ArtefactError)


def test_export_error_is_artefact():
    from kuaa.exporters.catalog import ExportError

    assert issubclass(ExportError, ArtefactError)


def test_upload_rejected_is_user_input_class():
    from kuaa.search.types import UploadRejected as CoreUploadRejected

    assert issubclass(CoreUploadRejected, UserInputError)
