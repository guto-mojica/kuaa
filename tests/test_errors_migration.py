"""Existing exceptions must inherit the F2 taxonomy (back-compat preserved)."""
from __future__ import annotations

from cinemateca.errors import (
    ArtefactError,
    RetrievalError,
    UserInputError,
)


def test_domain_error_is_user_input():
    from cinemateca.domain import DomainError

    assert issubclass(DomainError, UserInputError)


def test_dataset_error_is_user_input():
    from cinemateca.eval.datasets import DatasetError

    assert issubclass(DatasetError, UserInputError)


def test_eval_error_is_retrieval():
    from cinemateca.eval.retrieval import EvalError

    assert issubclass(EvalError, RetrievalError)


def test_annotation_stats_error_is_artefact():
    from cinemateca.eval.annotations import AnnotationStatsError

    assert issubclass(AnnotationStatsError, ArtefactError)


def test_export_error_is_artefact():
    from cinemateca.exporters.catalog import ExportError

    assert issubclass(ExportError, ArtefactError)


def test_upload_rejected_is_single_user_input_class():
    from api.services._search_image import UploadRejected as ApiUploadRejected
    from cinemateca.search.types import UploadRejected as CoreUploadRejected

    assert ApiUploadRejected is CoreUploadRejected  # de-duplicated
    assert issubclass(CoreUploadRejected, UserInputError)
