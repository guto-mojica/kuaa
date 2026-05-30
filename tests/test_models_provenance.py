"""C10 — describe_batch checkpoint is typed; each active backend has a ModelCard."""

from __future__ import annotations

import inspect

from cinemateca.models import base as base_mod


def test_describe_batch_checkpoint_is_typed_not_list_dict() -> None:
    sig = inspect.signature(base_mod.SceneDescriber.describe_batch)
    ann = sig.parameters["existing_results"].annotation
    # No longer the bare ``list[dict] | None`` — references the typed record.
    assert "SceneDescriptionRecord" in str(ann)


def test_scene_description_record_has_core_keys() -> None:
    keys = base_mod.SceneDescriptionRecord.__annotations__
    assert "scene_id" in keys
    assert "description" in keys


def test_registry_returns_model_card_for_active_roles() -> None:
    from cinemateca.config import load_config
    from cinemateca.models.registry import model_card

    cfg = load_config(ensure_dirs=False)
    card = model_card(cfg, "image_embedder")
    assert card.role == "image_embedder"
    assert card.model_id  # non-empty model id
