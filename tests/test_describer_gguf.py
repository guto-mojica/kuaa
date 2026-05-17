"""GGUF describer unit tests — llama-cpp fully mocked, hermetic."""
from __future__ import annotations

import pandas as pd

from cinemateca.models.base import SceneDescriber
from cinemateca.models.describer._common import PROMPTS


class _FakeLlama:
    """Stand-in for the llama_cpp.Llama wrapped by the backend."""

    def __init__(self):
        self.calls = 0

    def answer(self, prompt: str) -> str:
        self.calls += 1
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
    from cinemateca.models.describer import gguf

    fake = _FakeLlama()
    monkeypatch.setattr(
        gguf.MoondreamGGUFDescriber, "_answer",
        lambda self, image_path, prompt, max_tokens: fake.answer(prompt),
    )
    monkeypatch.setattr(
        gguf.MoondreamGGUFDescriber, "_load_model", lambda self: None,
    )
    return gguf.MoondreamGGUFDescriber(), fake


def test_gguf_describer_conforms(monkeypatch):
    backend, _ = _backend_with_fake(monkeypatch)
    assert isinstance(backend, SceneDescriber)


def test_describe_single_builds_metadata(monkeypatch):
    backend, _ = _backend_with_fake(monkeypatch)
    meta = backend.describe("frame.jpg")
    assert meta["location"] == "exterior"
    assert meta["time_of_day"] == "dia"
    assert meta["num_people"] == 2
    assert "tree" in meta["objects"]
    assert isinstance(meta["tags"], list) and meta["tags"]


def test_describe_batch_resume_excludes_error_rows(monkeypatch):
    """Regression: error rows must NOT count as processed (the resume bug)."""
    backend, fake = _backend_with_fake(monkeypatch)
    df = pd.DataFrame([
        {"filepath": "a.jpg", "scene_id": 1},
        {"filepath": "b.jpg", "scene_id": 2},
    ])
    existing = [{"scene_id": 1, "error": "boom", "tags": [], "objects": []}]
    out = backend.describe_batch(df, existing_results=existing)
    ids = sorted(r["scene_id"] for r in out)
    assert ids == [1, 2]
    assert any("error" not in r and r["scene_id"] == 1 for r in out)


def test_describe_batch_resume_preserves_good_rows(monkeypatch):
    """Good existing rows must be skipped (not reprocessed) and preserved."""
    backend, fake = _backend_with_fake(monkeypatch)
    df = pd.DataFrame([
        {"filepath": "a.jpg", "scene_id": 1},
        {"filepath": "b.jpg", "scene_id": 2},
    ])
    good_row = {
        "scene_id": 1,
        "description": "prior good",
        "tags": ["exterior"],
        "objects": [],
    }
    calls_before = fake.calls
    out = backend.describe_batch(df, existing_results=[good_row])
    # Only scene 2 was queried (len(PROMPTS) answers); scene 1 was skipped.
    assert fake.calls == calls_before + len(PROMPTS)
    ids = sorted(r["scene_id"] for r in out)
    assert ids == [1, 2]
    # The exact prior good row object is preserved, not rebuilt.
    assert any(
        r["scene_id"] == 1 and r.get("description") == "prior good" for r in out
    )
