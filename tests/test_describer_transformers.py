"""Transformers Moondream2 describer unit tests — model fully mocked, hermetic.

Mirrors tests/test_describer_gguf.py: never loads real weights, never
downloads, never touches repo data/.
"""

from __future__ import annotations

import pandas as pd

from cinemateca.models.base import SceneDescriber
from cinemateca.models.describer._common import PROMPTS


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


class _FakeAnswerer:
    """Call counter for the patched _answer method."""

    def __init__(self):
        self.calls = 0

    def answer(self, prompt: str) -> str:
        self.calls += 1
        return _answer_for(prompt)


def _backend_with_fake(monkeypatch):
    from cinemateca.models.describer import transformers_hf

    fake = _FakeAnswerer()
    monkeypatch.setattr(
        transformers_hf.MoondreamTransformersDescriber,
        "_load_model",
        lambda self: None,
    )
    monkeypatch.setattr(
        transformers_hf.MoondreamTransformersDescriber,
        "_answer",
        lambda self, image_path, prompt, max_tokens: fake.answer(prompt),
    )
    return transformers_hf.MoondreamTransformersDescriber(), fake


def test_transformers_describer_conforms(monkeypatch):
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
    backend, _ = _backend_with_fake(monkeypatch)
    df = pd.DataFrame(
        [
            {"filepath": "a.jpg", "scene_id": 1},
            {"filepath": "b.jpg", "scene_id": 2},
        ]
    )
    existing = [{"scene_id": 1, "error": "boom", "tags": [], "objects": []}]
    out = backend.describe_batch(df, existing_results=existing)
    ids = sorted(r["scene_id"] for r in out)
    assert ids == [1, 2]
    assert any("error" not in r and r["scene_id"] == 1 for r in out)


def test_describe_batch_resume_preserves_good_rows(monkeypatch):
    """Good existing rows must be skipped (not reprocessed) and preserved."""
    backend, fake = _backend_with_fake(monkeypatch)
    df = pd.DataFrame(
        [
            {"filepath": "a.jpg", "scene_id": 1},
            {"filepath": "b.jpg", "scene_id": 2},
        ]
    )
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
    assert any(r["scene_id"] == 1 and r.get("description") == "prior good" for r in out)


def _patch_cuda(monkeypatch, available: bool):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)


def test_warn_if_cpu_torch_warns_when_gpu_present_but_cpu_build(monkeypatch, caplog):
    """NVIDIA GPU present + CPU-only torch → loud WARNING."""
    from cinemateca.models.describer import transformers_hf

    backend = transformers_hf.MoondreamTransformersDescriber()
    monkeypatch.setattr(transformers_hf.shutil, "which", lambda _n: "/usr/bin/nvidia-smi")
    _patch_cuda(monkeypatch, available=False)
    with caplog.at_level("WARNING"):
        backend._warn_if_cpu_torch()
    assert any("CPU-only" in r.message for r in caplog.records)


def test_warn_if_cpu_torch_silent_when_cuda_available(monkeypatch, caplog):
    """A genuine CUDA torch build must NOT warn."""
    from cinemateca.models.describer import transformers_hf

    backend = transformers_hf.MoondreamTransformersDescriber()
    monkeypatch.setattr(transformers_hf.shutil, "which", lambda _n: "/usr/bin/nvidia-smi")
    _patch_cuda(monkeypatch, available=True)
    with caplog.at_level("WARNING"):
        backend._warn_if_cpu_torch()
    assert not caplog.records


def test_warn_if_cpu_torch_silent_without_nvidia_gpu(monkeypatch, caplog):
    """No nvidia-smi → CPU is expected, stay silent."""
    from cinemateca.models.describer import transformers_hf

    backend = transformers_hf.MoondreamTransformersDescriber()
    monkeypatch.setattr(transformers_hf.shutil, "which", lambda _n: None)
    _patch_cuda(monkeypatch, available=False)
    with caplog.at_level("WARNING"):
        backend._warn_if_cpu_torch()
    assert not caplog.records


def test_encode_called_once_per_frame(monkeypatch):
    """describe() runs len(PROMPTS) prompts but encodes the image once."""
    from cinemateca.models.describer import transformers_hf

    backend = transformers_hf.MoondreamTransformersDescriber()

    calls = {"encode": 0}

    class _FakeModel:
        def encode_image(self, img):
            calls["encode"] += 1
            return object()

        def answer_question(self, enc, prompt, tok, max_new_tokens):
            return _answer_for(prompt)

    def _fake_load(self):
        self._model = _FakeModel()
        self._tokenizer = object()

    monkeypatch.setattr(transformers_hf.MoondreamTransformersDescriber, "_load_model", _fake_load)
    # Make PIL Image.open/convert/resize a no-op-ish pass-through.
    import PIL.Image as _PILImage

    class _StubImg:
        def convert(self, _m):
            return self

        def resize(self, _s, _r):
            return self

    monkeypatch.setattr(_PILImage, "open", lambda _p: _StubImg())

    meta = backend.describe("frame.jpg")
    assert meta["num_people"] == 2
    assert calls["encode"] == 1, f"encoded {calls['encode']}x, expected 1"


def test_encode_called_once_per_frame_via_describe_batch(monkeypatch):
    """describe_batch encodes each frame once across all prompts (cache hit)."""
    from cinemateca.models.describer import transformers_hf

    backend = transformers_hf.MoondreamTransformersDescriber()
    calls = {"encode": 0}

    class _FakeModel:
        def encode_image(self, img):
            calls["encode"] += 1
            return object()

        def answer_question(self, enc, prompt, tok, max_new_tokens):
            return _answer_for(prompt)

    def _fake_load(self):
        self._model = _FakeModel()
        self._tokenizer = object()

    monkeypatch.setattr(
        transformers_hf.MoondreamTransformersDescriber, "_load_model", _fake_load
    )
    import PIL.Image as _PILImage

    class _StubImg:
        def convert(self, _m):
            return self

        def resize(self, _s, _r):
            return self

    monkeypatch.setattr(_PILImage, "open", lambda _p: _StubImg())

    df = pd.DataFrame([{"filepath": "only.jpg", "scene_id": 1}])
    out = backend.describe_batch(df)
    assert len(out) == 1 and out[0]["scene_id"] == 1
    assert calls["encode"] == 1, f"encoded {calls['encode']}x, expected 1"


def test_registry_returns_transformers_backend(monkeypatch):
    """registry.get_scene_describer resolves 'moondream_transformers'."""
    from cinemateca.models import registry
    from cinemateca.models.describer import transformers_hf

    monkeypatch.setattr(
        transformers_hf.MoondreamTransformersDescriber, "_load_model",
        lambda self: None,
    )

    class _Models:
        scene_describer = "moondream_transformers"

    class _Cfg:
        models = _Models()
        llm = None  # exercises the cfg-without-llm default branch

    backend = registry.get_scene_describer(_Cfg(), device=None)
    assert isinstance(backend, transformers_hf.MoondreamTransformersDescriber)


def test_registry_rejects_unknown_describer():
    from cinemateca.models import registry

    class _Models:
        scene_describer = "nope"

    class _Cfg:
        models = _Models()

    try:
        registry.get_scene_describer(_Cfg(), device=None)
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "nope" in str(e)
