# Protocol Option — Pluggable Model Backends

**Status:** Steps 1–2 implemented 2026-05-17 (pluggable Protocol + registry;
describer on keyless Moondream GGUF). Default describer reverted to
HF-transformers Moondream 2 on 2026-05-18 (deployment ease; GGUF stays
opt-in). Step 4 (ONNX) remains deferred. The Protocol set was later extended
with `clip_mclip` (M-CLIP), a second multilingual backend for the existing
`ImageEmbedder` Protocol — additive, no rewrites to the interface. `clip_mclip`
ships in the tree as an alternate/fallback backend; `siglip_multilingual`
is the shipped multilingual default (see `docs/MIGRATIONS.md`). Two further
backends were prototyped under this same Protocol and later removed
entirely from `main` to keep the current 0.8.0rc1 surface focused: a
`Transcriber` (faster-whisper) Protocol and an `AudioEmbedder` (LAION-CLAP)
Protocol for audio search plus CLIP×CLAP fusion. Neither exists in the
current tree; both were deliberately dropped rather than left half-wired.
**Context:** Decided during v0.3.0 stabilisation, before any model swaps begin.

---

## Release goal: standalone binary under 500 MB

The institution deployment target is a self-contained binary — no Python install,
no pip, no venv. The full install today is approximately:

| Component | Size |
|---|---|
| PyTorch + torchvision | ~650 MB |
| open-clip-torch | ~100 MB |
| transformers | ~100 MB |
| facenet-pytorch, ultralytics, etc. | ~150 MB |
| Python runtime + FastAPI + everything else | ~100 MB |
| **Total binary** | **~1.1 GB** |

Model weights are separate and stay separate regardless of approach (~2.4 GB,
downloaded on first run): CLIP ViT-B/32 ~400 MB, Moondream 2 ~2 GB, YOLOv8n ~6 MB.

The 500 MB target requires eliminating PyTorch from the binary entirely.
A language rewrite (Rust, Go) was considered and rejected — the frontend (HTMX,
Jinja2, Babel i18n) represents significant investment and the bottleneck is PyTorch
specifically, not Python.

### ONNX Runtime path

Replace PyTorch with ONNX Runtime (~35 MB) for CLIP, YOLOv8 and MTCNN, and replace
`transformers` with `llama-cpp-python` (~30 MB, pre-built wheels) for Moondream.

| Removed | Added | Size delta |
|---|---|---|
| `torch`, `torchvision` | `onnxruntime` | −615 MB |
| `open-clip-torch` | CLIP `.onnx` files (weights only, already counted) | −100 MB |
| `transformers` | `llama-cpp-python` | −70 MB |
| `facenet-pytorch` | ONNX MTCNN or OpenCV DNN (already a dep) | −50 MB |

**Estimated binary after migration: ~200–250 MB.** Well under target.

### On fine-tuning and model size

Fine-tuning does not reduce binary size — a fine-tuned model has the same
architecture and the same weight file size as the base model. What shrinks weights:

- **Quantization** (INT8/INT4) — cuts weight files 2–4×. Moondream GGUF is already
  quantized. Apply this at the weights level, independently of the binary approach.
- **Knowledge distillation** — train a smaller student. High effort, quality tradeoff.
- **Pruning** — hard to apply cleanly to transformers.

Quantization is the right lever for weight size. The Protocol design below is the
right lever for swapping in quantized or fine-tuned variants without touching the pipeline.

### One-time model export (pre-release step)

Before building the binary, each PyTorch model must be exported once:

- **CLIP**: `open_clip` provides export scripts → two `.onnx` files (image + text encoder).
- **YOLOv8**: `YOLO('yolov8n.pt').export(format='onnx')` — one command, native Ultralytics support.
- **MTCNN**: Community ONNX exports available; or swap for OpenCV DNN face detector (no new dep).
- **Moondream 2**: No official ONNX export. Must use GGUF via `llama-cpp-python`.
  Download `moondream-2b.gguf` from HuggingFace.

---

## Problem

Every model in `src/kuaa/` is hardwired to one backend:

- `CLIPEmbedder` → `open_clip` (PyTorch)
- `FaceDetector` → `facenet_pytorch` (PyTorch)
- `ObjectDetector` → `ultralytics` (PyTorch)
- `LLMDescriber` → `transformers` / Moondream 2 (PyTorch)

Swapping a model means editing the class. Migrating to ONNX means rewriting the class.
Each model change is surgery on production code.

With fine-tuned models and an ONNX migration both on the horizon, every change
will touch the same files more than once. That is the damage this design prevents.

---

## Design

Define a **typed interface (Protocol) per model role**. Concrete backends implement
the interface. The pipeline never imports a backend directly — it asks a registry,
which reads `config.yaml` and returns the right implementation.

This is the Strategy pattern applied per model role. It is inspired by ComfyUI's
pluggable node philosophy, adapted to a fixed pipeline (scenes → visual → embeddings
→ llm) rather than a visual graph.

**Adding a new model = one new file + one config line. Zero changes to the pipeline.**

---

## `src/kuaa/models/base.py` — The Interfaces

```python
"""
kuaa.models.base
~~~~~~~~~~~~~~~~~~~~~~
Protocol definitions for every model role in the pipeline.

Concrete backends (openclip, onnx, gguf, …) implement these interfaces.
The pipeline imports only from here — never from a concrete backend.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Protocol, runtime_checkable

import numpy as np
import pandas as pd

# ── Embedding ────────────────────────────────────────────────────────────────

@runtime_checkable
class ImageEmbedder(Protocol):
    """Encodes images into L2-normalised float32 vectors."""

    def encode_images(self, image_paths: List[Path]) -> np.ndarray:
        """Returns (N, D) float32 array, L2-normalised."""
        ...

    def encode_text(self, text: str) -> np.ndarray:
        """Returns (D,) float32 vector, L2-normalised.

        Text and image embeddings share the same space (CLIP contract).
        """
        ...

    def encode_image_single(self, image_path: Path) -> np.ndarray:
        """Returns (D,) float32 vector for one image (used in image search)."""
        ...

# ── Face detection ───────────────────────────────────────────────────────────

@runtime_checkable
class FaceDetector(Protocol):
    """Detects human faces in keyframe images."""

    def detect(self, image_path: Path) -> dict:
        """Returns {"num_faces": int, "faces": [{"bbox", "confidence", ...}]}."""
        ...

    def detect_batch(self, image_paths: List[Path]) -> List[dict]:
        """Returns one result dict per path, same order as input."""
        ...

# ── Object detection ─────────────────────────────────────────────────────────

@runtime_checkable
class ObjectDetector(Protocol):
    """Detects objects in keyframe images."""

    def detect(self, image_path: Path) -> dict:
        """Returns {"num_objects": int, "objects": [...], "class_counts": {...}}."""
        ...

    def detect_batch(self, image_paths: List[Path]) -> List[dict]:
        ...

# ── Scene description (VLM) ──────────────────────────────────────────────────

@runtime_checkable
class SceneDescriber(Protocol):
    """Generates natural-language metadata for a keyframe using a VLM."""

    def describe(self, image_path: Path) -> dict:
        """
        Returns a metadata dict with at minimum:
            description   : str
            location      : "interior" | "exterior" | "desconhecido"
            time_of_day   : "dia" | "noite" | "desconhecido"
            num_people    : int  (-1 = vague/many)
            objects       : List[str]
            tags          : List[str]  (kebab-case, auto-generated)
        """
        ...

    def describe_batch(
        self,
        keyframes_df: pd.DataFrame,
        existing_results: list | None = None,
        checkpoint_path: Path | None = None,
    ) -> list[dict]:
        """Processes all rows in keyframes_df, supports resume via existing_results."""
        ...

# ── Environment classification ───────────────────────────────────────────────

@runtime_checkable
class EnvironmentClassifier(Protocol):
    """Heuristic or model-based environment classification."""

    def classify(self, image_path: Path) -> dict:
        """Returns {"time_of_day", "brightness_score", "location", "edge_density"}."""
        ...

    def classify_batch(self, image_paths: List[Path]) -> List[dict]:
        ...
```

> **Current state:** all five Protocol roles
> above — `ImageEmbedder`, `FaceDetector`, `ObjectDetector`, `SceneDescriber`,
> `EnvironmentClassifier` — still exist in `src/kuaa/models/base.py` and are
> unchanged in intent. The live file has grown beyond this original sketch:
> `ImageEmbedder` and `SceneDescriber` each gained a `save(...)` method so
> backends persist their own output format, and `SceneDescriber` now types its
> batch output as a `SceneDescriptionRecord` `TypedDict` instead of a bare
> `dict`. None of that changes the shape of this document's argument — it is
> exactly the kind of additive, interface-preserving growth the Protocol was
> designed to absorb.

---

## `src/kuaa/models/registry.py` — The Factory

```python
"""
kuaa.models.registry
~~~~~~~~~~~~~~~~~~~~~~~~~~
Reads the [models] section of config.yaml and returns concrete
implementations. The pipeline imports from here, not from backends.
"""
from __future__ import annotations

from functools import lru_cache
from kuaa.models.base import (
    ImageEmbedder, FaceDetector, ObjectDetector,
    SceneDescriber, EnvironmentClassifier,
)

def get_image_embedder(cfg) -> ImageEmbedder:
    name = cfg.models.image_embedder  # e.g. "clip_openclip"
    if name == "clip_openclip":
        from kuaa.models.clip.openclip import OpenClipEmbedder
        return OpenClipEmbedder(cfg)
    if name == "clip_onnx":
        from kuaa.models.clip.onnx import OnnxClipEmbedder
        return OnnxClipEmbedder(cfg)
    raise ValueError(f"Unknown image_embedder: {name!r}")

def get_face_detector(cfg) -> FaceDetector:
    name = cfg.models.face_detector
    if name == "mtcnn_pytorch":
        from kuaa.models.face.mtcnn import MTCNNDetector
        return MTCNNDetector(cfg)
    if name == "mtcnn_onnx":
        from kuaa.models.face.onnx import OnnxMTCNNDetector
        return OnnxMTCNNDetector(cfg)
    if name == "opencv_dnn":
        from kuaa.models.face.opencv_dnn import OpenCVDNNDetector
        return OpenCVDNNDetector(cfg)
    raise ValueError(f"Unknown face_detector: {name!r}")

def get_object_detector(cfg) -> ObjectDetector:
    name = cfg.models.object_detector
    if name == "yolov8":
        from kuaa.models.objects.yolov8 import YOLOv8Detector
        return YOLOv8Detector(cfg)
    if name == "yolov8_onnx":
        from kuaa.models.objects.onnx import OnnxYOLODetector
        return OnnxYOLODetector(cfg)
    raise ValueError(f"Unknown object_detector: {name!r}")

def get_scene_describer(cfg) -> SceneDescriber:
    name = cfg.models.scene_describer
    if name == "moondream_transformers":
        # DEFAULT — vikhyatk/moondream2 @2025-01-09 via HF transformers.
        # GPU acceleration through the prebuilt PyTorch CUDA/MPS wheel;
        # no source build required. Requires transformers>=4.44,<5
        # (transformers 5 hard-fails for every moondream2 revision).
        from kuaa.models.describer.transformers_hf import MoondreamTransformersDescriber
        return MoondreamTransformersDescriber(cfg)
    if name == "moondream_gguf":
        # OPT-IN — vikhyatk/moondream2 GGUF via llama-cpp-python.
        # GPU requires a source build of llama-cpp-python with CUDA
        # (see docs/GPU_LLAMA_CPP_CUDA_BUILD.md). Use when the
        # transformers wheel is too large or GGUF inference is preferred.
        from kuaa.models.describer.gguf import GGUFDescriber
        return GGUFDescriber(cfg)
    raise ValueError(f"Unknown scene_describer: {name!r}")
```

> **Current state:** the live `registry.py` follows exactly this
> factory-per-role shape, with three differences worth flagging rather than
> silently absorbing. First, `get_image_embedder` caches instances by
> `(backend_name, device)` so repeated calls at query time don't reconstruct
> weights. Second, it has a third `image_embedder` branch, `clip_mclip`
> (M-CLIP, see `docs/MIGRATIONS.md` and `docs/MODEL_INVENTORY.md`), alongside
> `clip_openclip` and `siglip_multilingual` — there is no `clip_onnx` branch
> yet; ONNX backends are still the deferred Step 4 below, not yet built.
> Third, the registry now exposes a config-aware `model_card(settings, role)`
> accessor on top of the `get_*` factories, so provenance (model id, license,
> revision) is resolved from the *active* backend rather than guessed.
> `moondream_pytorch` (a legacy non-transformers PyTorch backend once
> prototyped alongside GGUF) is gone from the live registry entirely — only
> `moondream_transformers` and `moondream_gguf` remain, matching the two
> rows in `docs/MODEL_INVENTORY.md`.

---

## `config/default.yaml` addition

```yaml
models:
  image_embedder: clip_openclip      # clip_openclip | clip_onnx
  face_detector: mtcnn_pytorch       # mtcnn_pytorch | mtcnn_onnx | opencv_dnn
  object_detector: yolov8            # yolov8 | yolov8_onnx
  scene_describer: moondream_transformers  # moondream_transformers (default) | moondream_gguf (opt-in)
  environment_classifier: opencv_heuristic  # only one impl today
```

Override in `config/local.yaml` to test a new backend without touching code.
The live default has since moved: `image_embedder` now defaults to
`siglip_multilingual`, not `clip_openclip` — see `docs/MIGRATIONS.md` for the
rollout that changed it. The shape of the config block (one selector per
role, one line each) is unchanged.

---

## Repository layout after restructure

```
src/kuaa/
  models/
    __init__.py
    base.py              ← Protocols (this document)
    registry.py          ← factory functions
    clip/
      openclip.py        ← current CLIPEmbedder, renamed + interface-compliant
      onnx.py            ← future ONNX Runtime backend
    face/
      mtcnn.py           ← current FaceDetector, renamed
      onnx.py            ← future
      opencv_dnn.py      ← free (OpenCV already a dep)
    objects/
      yolov8.py          ← current ObjectDetector, renamed
      onnx.py            ← future
    describer/
      transformers_hf.py  ← DEFAULT: HF transformers, GPU via prebuilt wheel
      moondream.py               ← legacy moondream_pytorch backend
      gguf.py                    ← opt-in: llama-cpp-python, GPU requires source build
  embeddings.py          ← SemanticSearch stays here (pure numpy, no model)
  scene_detector.py      ← unchanged (PySceneDetect, no PyTorch)
  annotator.py           ← unchanged
  pipeline.py            ← imports from registry, not from backends
  device.py              ← becomes provider selector for ONNX, or removed
```

**What does NOT move:** `SemanticSearch` (pure numpy dot-product, not a model),
`scene_detector.py` (PySceneDetect/OpenCV), `annotator.py`, `library.py`.

> **Current state:** `base.py`, `registry.py`, `embeddings.py`,
> `scene_detector.py`, `pipeline.py`, and `device.py` all exist exactly
> where this layout plan puts them.
> `models/clip/openclip.py`, `models/face/mtcnn.py`, `models/objects/yolov8.py`,
> and `models/describer/transformers_hf.py` + `models/describer/gguf.py` also
> exist as planned. Three things drifted from this plan and are worth naming
> rather than leaving as a silent inaccuracy:
>
> - `describer/moondream.py` (the legacy `moondream_pytorch` backend) does
>   not exist — that backend was dropped outright rather than kept around as
>   a legacy file; `registry.get_scene_describer` only recognises
>   `moondream_transformers` and `moondream_gguf`.
> - `annotator.py` does not exist as a single file. Manual tag curation now
>   lives under `src/kuaa/annotations/` as a package (`descriptions.py`,
>   `io.py`, `overrides.py`, `scenes.py`), and `library.py` similarly became
>   the `src/kuaa/library/` package. Both are still outside `models/`, which
>   is the point this section was making.
> - None of the `onnx.py` / `opencv_dnn.py` future placeholders exist yet —
>   Step 4 below is still not started. The one additional backend that *did*
>   ship beyond this plan is `models/clip/mclip.py` (M-CLIP), which slots into
>   the `clip/` subfolder exactly as this layout anticipates for alternate
>   `ImageEmbedder` backends.

---

## Migration path

This restructure is **not** a rewrite. Each concrete file is the current class,
moved and renamed to implement the Protocol. Logic changes only when a new backend
is added.

```
Step 1  Define Protocols in base.py (this doc).
        No logic changes anywhere. Existing classes implicitly satisfy the Protocol
        via structural subtyping — Python checks duck-typing, no explicit inheritance.

Step 2  Add registry.py pointing to current implementations under models/.
        Update pipeline.py to call registry.get_*() instead of importing directly.
        Tests still pass; behaviour unchanged.

Step 3  (When a model swap is needed) Add one new file under the right subfolder.
        Update config.yaml. Done.

Step 4  (Pre-release) Add ONNX backends as new files. Switch config to clip_onnx,
        yolov8_onnx, etc. Drop PyTorch from dependencies. Build binary.
```

---

## Timing

Do Step 1 + Step 2 **before** any model changes. The restructure is low-risk
(logic-preserving), but it makes every subsequent model change additive instead
of surgical. If model changes come before the Protocol layer, each swap will
touch shared code under pressure.

Do Step 4 (ONNX) only when a release build is the explicit goal — not before,
because ONNX requires one-time model exports and changes the dependency tree.

---

## Relation to ONNX migration

The Protocol layer is the prerequisite for an eventual ONNX migration: once
`registry.py` exists, adding an ONNX backend is one new file per role, not a
migration of existing classes. The ONNX migration cost drops significantly
once this structure is in place — which is why Steps 1–2 were done first and
Step 4 is still explicitly deferred rather than attempted piecemeal.
