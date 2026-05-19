"""Transformers Moondream2 describer unit tests — model fully mocked, hermetic.

Mirrors tests/test_describer_gguf.py: never loads real weights, never
downloads, never touches repo data/.
"""

from __future__ import annotations

from cinemateca.models.base import SceneDescriber


def _answer_for(prompt: str) -> str:
    if "indoors or outdoors" in prompt:
        return "outdoor"
    if "time of day" in prompt:
        return "day"
    if "How many people" in prompt:
        return "2 people talking"
    if "notable objects" in prompt:
        return "tree, fence"
    if "setting in" in prompt:
        return "rural field"
    return "A man stands in a field."


def _backend_with_fake(monkeypatch):
    from cinemateca.models.describer import transformers_hf

    monkeypatch.setattr(
        transformers_hf.MoondreamTransformersDescriber,
        "_load_model",
        lambda self: None,
    )
    monkeypatch.setattr(
        transformers_hf.MoondreamTransformersDescriber,
        "_answer",
        lambda self, image_path, prompt, max_tokens: _answer_for(prompt),
    )
    return transformers_hf.MoondreamTransformersDescriber()


def test_transformers_describer_conforms(monkeypatch):
    backend = _backend_with_fake(monkeypatch)
    assert isinstance(backend, SceneDescriber)


def test_describe_single_builds_metadata(monkeypatch):
    backend = _backend_with_fake(monkeypatch)
    meta = backend.describe("frame.jpg")
    assert meta["location"] == "exterior"
    assert meta["time_of_day"] == "dia"
    assert meta["num_people"] == 2
    assert "tree" in meta["objects"]
    assert isinstance(meta["tags"], list) and meta["tags"]
