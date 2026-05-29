"""G0 contract: every pinned Phase-0 public name imports and behaves.

This is the single guard the downstream workstream plans rely on — if a
name here is renamed, those plans break, so it must stay green.
"""

from __future__ import annotations


def test_config_public_surface():
    from cinemateca.config import Settings, load_config  # noqa: F401

    cfg = load_config(ensure_dirs=False)
    assert isinstance(cfg, Settings)


def test_errors_public_surface():
    from cinemateca.errors import (  # noqa: F401
        ArtefactError,
        CinematecaError,
        ConfigError,
        IndexMissing,
        ModelError,
        PipelineError,
        RetrievalError,
        UserInputError,
    )

    assert issubclass(ConfigError, CinematecaError)


def test_reproducibility_public_surface():
    from cinemateca.reproducibility import seed_everything

    seed_everything(0)  # callable, no raise


def test_snapshot_public_surface():
    from tests._snapshot import assert_snapshot  # noqa: F401

    assert callable(assert_snapshot)


def test_timing_public_surface():
    from cinemateca.timing import timed

    with timed("x") as t:
        pass
    assert t.elapsed_ms >= 0.0


def test_manifest_and_registry_card_surface():
    from cinemateca.config import load_config
    from cinemateca.models.manifest import ModelCard  # noqa: F401
    from cinemateca.models.registry import model_card

    cfg = load_config(ensure_dirs=False)
    assert model_card(cfg, "image_embedder").role == "image_embedder"


def test_middleware_module_exposes_request_id_header():
    from api.middleware import REQUEST_ID_HEADER, RequestContextMiddleware  # noqa: F401

    assert REQUEST_ID_HEADER == "X-Request-ID"
