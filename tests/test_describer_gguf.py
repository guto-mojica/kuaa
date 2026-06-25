"""GGUF describer unit tests — llama-cpp fully mocked, hermetic."""

from __future__ import annotations

import pandas as pd
import pytest

from kuaa.models.base import SceneDescriber
from kuaa.models.describer._common import PROMPTS


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
    from kuaa.models.describer import gguf

    fake = _FakeLlama()
    monkeypatch.setattr(
        gguf.MoondreamGGUFDescriber,
        "_answer",
        lambda self, image_path, prompt, max_tokens: fake.answer(prompt),
    )
    monkeypatch.setattr(
        gguf.MoondreamGGUFDescriber,
        "_load_model",
        lambda self: None,
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
    # The exact prior good row object is preserved, not rebuilt.
    assert any(r["scene_id"] == 1 and r.get("description") == "prior good" for r in out)


def _force_offload(monkeypatch, value: bool):
    """Hermetically pin llama_cpp's GPU-offload capability flag."""
    core = pytest.importorskip("llama_cpp.llama_cpp")
    monkeypatch.setattr(core, "llama_supports_gpu_offload", lambda: value)


def test_warn_if_cpu_build_warns_when_gpu_present_but_cpu_only(monkeypatch, caplog):
    """GPU requested + NVIDIA GPU present + CPU-only build → loud WARNING."""
    from kuaa.models.describer import gguf

    backend = gguf.MoondreamGGUFDescriber()
    backend.n_gpu_layers = -1
    monkeypatch.setattr(gguf.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")
    _force_offload(monkeypatch, False)

    with caplog.at_level("WARNING"):
        backend._warn_if_cpu_build()

    assert any(
        "CPU-only" in r.message and "GPU_LLAMA_CPP_CUDA_BUILD.md" in r.message
        for r in caplog.records
    )


def test_warn_if_cpu_build_silent_when_gpu_offload_available(monkeypatch, caplog):
    """A genuine CUDA build (offload available) must NOT warn."""
    from kuaa.models.describer import gguf

    backend = gguf.MoondreamGGUFDescriber()
    backend.n_gpu_layers = -1
    monkeypatch.setattr(gguf.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")
    _force_offload(monkeypatch, True)

    with caplog.at_level("WARNING"):
        backend._warn_if_cpu_build()

    assert not caplog.records


def test_warn_if_cpu_build_silent_without_nvidia_gpu(monkeypatch, caplog):
    """No nvidia-smi → CPU-only build is expected, stay silent."""
    from kuaa.models.describer import gguf

    backend = gguf.MoondreamGGUFDescriber()
    backend.n_gpu_layers = -1
    monkeypatch.setattr(gguf.shutil, "which", lambda _name: None)
    _force_offload(monkeypatch, False)

    with caplog.at_level("WARNING"):
        backend._warn_if_cpu_build()

    assert not caplog.records


def test_warn_if_cpu_build_silent_when_gpu_layers_zero(monkeypatch, caplog):
    """gpu_layers=0 means CPU was explicitly chosen → no warning."""
    from kuaa.models.describer import gguf

    backend = gguf.MoondreamGGUFDescriber()
    backend.n_gpu_layers = 0
    monkeypatch.setattr(gguf.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")
    _force_offload(monkeypatch, False)

    with caplog.at_level("WARNING"):
        backend._warn_if_cpu_build()

    assert not caplog.records


def test_gguf_init_defaults_when_llm_is_none():
    """Guard: cfg present but cfg.llm is None → hardcoded defaults, no AttributeError."""
    from kuaa.models.describer.gguf import MoondreamGGUFDescriber

    class _Cfg:
        llm = None

    backend = MoondreamGGUFDescriber(_Cfg())
    assert backend.checkpoint_interval == 25
    assert backend.descriptions_filename == "scene_descriptions.json"
    assert backend.tags_filename == "scene_tags.json"
    assert backend.process_limit is None
    assert backend.n_gpu_layers == -1
