# Model inventory

Status: public documentation draft.

> This inventory is rendered from the model provenance manifest
> (`src/kuaa/models/manifest.py`); the registry's `model_card(settings, role)` is the
> source of truth, so code and docs do not drift.

This document records the model roles, current backends, licenses, privacy
considerations, and replacement risks for KUAA. It is not legal
advice. Before institutional deployment, commercial deployment, or publication
of a packaged release, verify every license against the exact package and model
weights being distributed.

## Summary

| Role | Current backend | Package/model | Config key | Notes |
|---|---|---|---|---|
| Image/text embeddings | SigLIP 2 (multilingual) | `google/siglip2-large-patch16-256` (1024-dim) | `models.image_embedder: siglip_multilingual` | Default; shared multilingual text+image space. |
| Image/text embeddings (alternate) | OpenCLIP | `open_clip`, `ViT-B-32`, `openai` (512-dim) | `models.image_embedder: clip_openclip` | Alternative/legacy; per-film `.clip_openclip.npy` backups kept for rollback. |
| Image/text embeddings (unshipped fallback) | M-CLIP | `sentence-transformers`, `clip-ViT-B-32-multilingual-v1` (512-dim) | `models.image_embedder: clip_mclip` | Multilingual text encoder over the OpenCLIP ViT-B/32 image space; superseded by SigLIP 2 as the shipped multilingual default, kept as a fallback. |
| Face detection | MTCNN | `facenet-pytorch` | `models.face_detector: mtcnn_pytorch` | Detects face counts/boxes, not identity. |
| Object detection | YOLOv8 | `ultralytics`, `yolov8n.pt` | `models.object_detector: yolov8` | AGPL/Enterprise license concern. |
| Scene description | Moondream 2 | `vikhyatk/moondream2` via transformers | `models.scene_describer: moondream_transformers` | Default local VLM backend. |
| Scene description alternative | Moondream 2 GGUF | `vikhyatk/moondream2` GGUF via llama-cpp-python | `models.scene_describer: moondream_gguf` | Offline/keyless; GPU acceleration depends on local build. |
| Environment classification | OpenCV heuristic | local code | `models.environment_classifier: opencv_heuristic` | Brightness/edge heuristic, not learned. |

## Backend details

### Image/text embeddings

Purpose:

- Encode keyframe images into normalized vectors.
- Encode text queries into the same vector space.
- Support image-to-image search by encoding a reference image.

Implementation:

- `src/kuaa/models/clip/siglip_multilingual.py`
- `src/kuaa/models/clip/openclip.py`
- `src/kuaa/models/clip/mclip.py`
- `src/kuaa/embeddings.py`

Configured default:

- backend: `siglip_multilingual`
- model: `google/siglip2-large-patch16-256` (SigLIP 2, multilingual)
- embedding dim: 1024, L2-normalised, shared text+image space
- package: `transformers`

Alternate backend: OpenCLIP `ViT-B-32` / `openai` (512-dim, `open-clip-torch`); per-film
`.clip_openclip.npy` backups are preserved for rollback (see `docs/MIGRATIONS.md`).

Unshipped fallback backend: M-CLIP (`clip-ViT-B-32-multilingual-v1` via
`sentence-transformers`) overrides only `encode_text()` on top of the OpenCLIP
ViT-B/32 image encoder, so it shares the 512-dim OpenCLIP space and needs no
re-embedding. It has no manifest card by design — it is intentionally absent from
`CARDS` because it isn't the shipped multilingual default; `siglip_multilingual`
is. If M-CLIP is ever promoted to shipped status, a `clip_mclip` card should be
added to the manifest at that time.

License/source notes:

- OpenCLIP code is MIT-style licensed in the upstream repository.
- Pretrained weight licensing can vary by checkpoint. This project currently
  uses the `openai` checkpoint through OpenCLIP; verify the exact checkpoint
  terms before redistribution.

Operational notes:

- Model weights download on first load through the package's normal mechanism.
- Embeddings are stored locally in `data/library/<slug>/embeddings/keyframe_embeddings.npy`.
- Index mapping is stored locally in `data/library/<slug>/embeddings/index_mapping.json`.

Replacement candidates:

- ONNX CLIP backend for smaller packaged builds.
- Different OpenCLIP pretrained checkpoints for quality comparison.
- Domain-specific fine-tuned embedding model after evaluation identifies a need.

### Face detection

Purpose:

- Count visible faces in keyframes.
- Store face boxes and confidence where available.
- Support visual metadata and future filtering.

Implementation:

- `src/kuaa/models/face/mtcnn.py`

Current backend:

- `facenet-pytorch` MTCNN.

License/source notes:

- The upstream `facenet-pytorch` repository is MIT licensed.
- It includes pretrained model behavior; verify weight provenance before
  packaging for institutional distribution.

Operational notes:

- This project does not identify people by name.
- It should be described publicly as face detection/counting, not face
  recognition.

Replacement candidates:

- OpenCV DNN face detector.
- ONNX MTCNN or another permissive local face detector.
- Disable face detection entirely for sensitive collections.

### Object detection

Purpose:

- Detect common objects in keyframes.
- Store object classes, counts, boxes, and confidence.

Implementation:

- `src/kuaa/models/objects/yolov8.py`

Configured default:

- package: `ultralytics`
- model: `yolov8n.pt`
- confidence: `0.30`

License/source notes:

- Ultralytics documents YOLOv8 models as provided under AGPL-3.0 and Enterprise
  licenses.
- This is the most important license risk in the current model stack.
- For public open-source portfolio use, the AGPL issue should be disclosed.
- For institutional or commercial use, either comply with the relevant license
  obligations, obtain appropriate licensing, disable this backend, or replace it
  with a backend whose terms match the deployment.

Operational notes:

- YOLOv8 is useful for a quick object metadata pass, but generic COCO labels are
  not archive-specific.
- Historical film quality, black-and-white footage, unusual framing, and old
  objects can reduce accuracy.

Replacement candidates:

- YOLO exported to ONNX only if license terms are acceptable.
- RT-DETR/DETR-family backends with suitable licenses.
- Domain-specific object vocabulary through a future domain pack.
- Optional object detection disabled by default for a minimal permissive demo.

### Scene description

Purpose:

- Generate natural-language scene descriptions.
- Extract structured metadata fields such as location, setting, time of day,
  number of people, actions, objects, and tags.

Shared implementation:

- `src/kuaa/models/describer/_common.py`

Default backend:

- `src/kuaa/models/describer/transformers_hf.py`
- model id: `vikhyatk/moondream2`
- default revision in config: `2025-01-09`
- package: `transformers`
- requires `trust_remote_code=True`

Alternative backend:

- `src/kuaa/models/describer/gguf.py`
- model repo: `vikhyatk/moondream2`
- revision: `2025-01-09`
- files:
  - `moondream2-text-model-f16.gguf`
  - `moondream2-mmproj-f16.gguf`
- packages:
  - `llama-cpp-python`
  - `huggingface-hub`

License/source notes:

- The Hugging Face model card for `vikhyatk/moondream2` lists license
  `apache-2.0`.
- The model repository is updated over time; keep explicit revisions for
  reproducibility.

Operational notes:

- The default transformers backend can use CUDA or Apple MPS through PyTorch.
- The GGUF backend is keyless and offline after weights are cached, but GPU
  offload requires a compatible local `llama-cpp-python` build.
- VLM outputs are probabilistic and can be wrong. The annotation layer is part
  of the intended workflow.

Replacement candidates:

- Newer Moondream revisions after regression testing.
- Qwen/SmolVLM-style local VLM backends if licenses and dependencies fit.
- Domain-specific prompt packs before model fine-tuning.

### Environment classifier

Purpose:

- Estimate indoor/outdoor and day/night-style metadata.

Implementation:

- `src/kuaa/models/environment/opencv_heuristic.py`

Current backend:

- Local OpenCV heuristic based on brightness and edge density.

License/source notes:

- This is local project code using OpenCV.

Operational notes:

- This is approximate. It should not be presented as a trained classifier.
- Historical black-and-white footage can confuse brightness-based day/night
  heuristics.

Replacement candidates:

- Small trained classifier on archive-specific labels.
- Domain-pack-specific classifier behavior.
- A VLM-based classifier pass if latency is acceptable.

### Cross-encoder reranker

Purpose:

- Re-score text-search candidates with a cross-encoder after card enrichment.

Implementation:

- `kuaa.search.rerank`

Configured default:

- backend: `bge_reranker_v2_m3`
- model: `BAAI/bge-reranker-v2-m3`
- license: Apache-2.0

Status:

**Rerank ships OFF by default** (`retrieval.reranker.enabled: false` in
`config/default.yaml`, confirmed against the live config). `/api/search`
accepts and logs `reranker_enabled`; the cross-encoder (`BAAI/bge-reranker-v2-m3`,
in `kuaa.search.rerank`) is wired into `find()` and applies after enrichment
when opted in (`?reranker_enabled=true`). It is off by default because its effect
was **unmeasured** at the time of the decision — the proxy ablation reranked
empty descriptions, a since-fixed core-path bug — and its text-only design is
suspect on short captions; see `docs/RERANKER_DECISION.md` for the full
reasoning and current status. Image requests ignore the text reranker regardless.

Operational notes:

- Weights download on first rerank.

## Download and cache behavior

This project is offline-first during use, but not necessarily zero-network
during setup.

Expected first-run downloads may include:

- OpenCLIP or SigLIP 2 weights.
- YOLOv8 weights.
- MTCNN/facenet-pytorch weights.
- Moondream 2 weights through Hugging Face or GGUF downloads.
- Python packages during environment setup.

After dependencies and weights are installed/cached, the app is intended to run
locally without sending videos, keyframes, annotations, embeddings, or search
queries to external APIs.

## Public documentation rules

Use careful language:

- Say "local/offline after installation and model download", not "never touches
  the network."
- Say "face detection/counting", not "face recognition."
- Say "machine-generated metadata for human review", not "automatic ground truth."
- Say "YOLOv8 has AGPL/Enterprise licensing implications", not "safe for any use."
- Say "heuristic environment classifier", not "trained environment model."

## License summary

One place for every shipped model's license. This is not legal advice; verify
against the exact package version and weights before institutional/commercial
deployment. Rendered from the model manifest where available
(`src/kuaa/models/manifest.py`, `ModelCard.license`).

| Role | Model / package | License | Redistribution note |
|---|---|---|---|
| Image/text embeddings (default) | `google/siglip2-large-patch16-256` (SigLIP 2) | Apache-2.0 (Google SigLIP weights) | Verify checkpoint terms before redistributing weights. |
| Image/text embeddings (alternate) | OpenCLIP `ViT-B-32` / `openai` | OpenCLIP code MIT; checkpoint terms vary | The `openai` checkpoint terms apply; verify before redistribution. |
| Image/text embeddings (unshipped fallback) | M-CLIP `clip-ViT-B-32-multilingual-v1` via `sentence-transformers` | Apache-2.0 (sentence-transformers); checkpoint terms vary | Not shipped as a default; verify checkpoint terms before promoting. |
| Cross-encoder reranker | `BAAI/bge-reranker-v2-m3` | Apache-2.0 (BAAI) | Verify model card at pin time. |
| Scene description | `vikhyatk/moondream2` (Moondream 2) | Apache-2.0 (per HF model card) | Keep an explicit revision pin for reproducibility. |
| Object detection | Ultralytics YOLOv8 (`yolov8n.pt`) | **AGPL-3.0 / Enterprise** | **Highest-risk dependency.** Disclose for OSS; obtain Enterprise license, disable, or replace for institutional/commercial use. |
| Face detection | `facenet-pytorch` MTCNN | MIT | Detection/counting only, not recognition. Verify weight provenance. |
| Environment classifier | local OpenCV heuristic | project code (MIT) + OpenCV (Apache-2.0) | Heuristic, not a trained model. |

The **YOLOv8 AGPL-3.0** obligation is the one license risk that materially
affects how this project can be shipped. See the README license note and the
"Object detection" section above for the disable/replace options.

## Pre-release verification checklist

Before a public release or demo package:

- Verify exact model licenses for the package versions and weights being used.
- Decide whether YOLOv8 is included, disabled by default, or replaced.
- Record model revisions in a run manifest.
- Confirm model weights are not accidentally committed if they are too large or
  not intended for redistribution.
- Confirm demo data provenance and rights.
- Confirm the README does not imply formal privacy/compliance certification.

## Official source links

- OpenCLIP repository/license: https://github.com/mlfoundations/open_clip
- Moondream 2 model card: https://huggingface.co/vikhyatk/moondream2
- Ultralytics YOLOv8 docs: https://docs.ultralytics.com/models/yolov8
- facenet-pytorch repository/license: https://github.com/timesler/facenet-pytorch

## A note on `clip_mclip` (M-CLIP)

Every role and backend named above — `siglip_multilingual`, `clip_openclip`,
`moondream_transformers`, `moondream_gguf`, `yolov8`, `mtcnn_pytorch`,
`opencv_heuristic`, and `bge_reranker_v2_m3` — has a matching `ModelCard`
entry in `src/kuaa/models/manifest.py` with the same model id, license, and
dimension claimed here. One backend, `clip_mclip` (M-CLIP), has no manifest
card by design: per `src/kuaa/models/clip/mclip.py`, it's a fallback, not the
shipped multilingual default, and its absence from `CARDS` is intentional,
not an oversight.
