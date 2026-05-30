"""Pydantic v2 typed configuration schema (F1).

Every section of ``config/default.yaml`` is transcribed into a
:class:`_Section` subclass with ``extra="forbid"`` so an unknown key (a
typo, a removed field, a stale local override) fails loudly at load time
instead of silently being ignored. Selector fields (model-backend ids,
device names, scene detector) are ``Literal`` so an invalid value is a
schema error rather than a deep ``AttributeError`` at use time.

The root is a plain :class:`pydantic.BaseModel` (not ``BaseSettings``):
the configuration is YAML-file-driven, not environment-driven, and
``pydantic-settings`` is not a project dependency. Three thin shims on
``_Section`` (``get``) and :class:`Settings` (``to_dict``) reproduce the
public surface of the old ``_Namespace`` so the 22 existing
``load_config`` callers and ``run_manifest.config_snapshot`` keep working
unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class _Section(BaseModel):
    """Base for every config section: reject unknown keys, allow dot-access.

    ``get`` mirrors the old ``_Namespace.get`` so callers doing
    ``cfg.x.get('y', default)`` keep working; ``Path`` fields and nested
    models are left as-is.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict view of this section, ``Path`` objects preserved.

        Every section exposes this — not just the root — because
        :func:`cinemateca.run_manifest._coerce_jsonable` recurses via
        ``hasattr(value, 'to_dict')`` and embeds nested config nodes
        (e.g. ``cfg.models``) directly into the manifest. The old
        ``_Namespace`` provided ``to_dict`` on every node; preserving
        that contract keeps the manifest JSON byte-for-byte equivalent.
        ``mode='python'`` keeps ``Path`` objects (run_manifest coerces
        them to ``str``), matching the old behaviour.
        """
        return self.model_dump(mode="python")


CONFIG_VERSION = 1


# ── project ──────────────────────────────────────────────────────────────────
class ProjectCfg(_Section):
    name: str
    version: str


# ── domain ───────────────────────────────────────────────────────────────────
class DomainCfg(_Section):
    pack: str
    packs_dir: str
    # Optional explicit pack path. Absent from default.yaml (pack is selected
    # by name there), but ``cinemateca.domain.resolve_domain_pack_path`` reads
    # ``getattr(domain_cfg, "path", None)`` and a user/test override can set it
    # to point at a specific ``*.yaml`` pack, taking precedence over ``pack``.
    path: str | None = None


# ── paths ────────────────────────────────────────────────────────────────────
class PathsCfg(_Section):
    # Loader resolves these to absolute Paths before model construction.
    library_dir: Path
    data_dir: Path
    raw_dir: Path
    frames_dir: Path
    metadata_dir: Path
    embeddings_dir: Path
    models_dir: Path
    outputs_dir: Path
    logs_dir: Path


# ── hardware ─────────────────────────────────────────────────────────────────
class HardwareCfg(_Section):
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    force_cpu: bool = False


# ── frame_extraction ─────────────────────────────────────────────────────────
class FrameExtractionCfg(_Section):
    fps: int = 1
    target_height: int = 480
    quality: int = 2
    sample_duration: int | None = None


# ── scene_detection ──────────────────────────────────────────────────────────
class SceneDetectionCfg(_Section):
    detector: Literal["content", "adaptive"] = "adaptive"
    content_threshold: float = 27.0
    adaptive_threshold: float = 3.0
    min_scene_len: int = 15
    keyframes_per_scene: int = 3
    keyframe_height: int = 480


# ── visual_analysis (+ nested face/object/environment) ───────────────────────
class FaceDetectionCfg(_Section):
    enabled: bool = True
    min_face_size: int = 20
    thresholds: list[float] = [0.6, 0.7, 0.7]


class ObjectDetectionCfg(_Section):
    enabled: bool = True
    model: str = "yolov8n.pt"
    confidence: float = 0.30


class EnvironmentCfg(_Section):
    enabled: bool = True
    brightness_threshold: int = 100
    edge_density_threshold: float = 0.05


class VisualAnalysisCfg(_Section):
    face_detection: FaceDetectionCfg
    object_detection: ObjectDetectionCfg
    environment: EnvironmentCfg


# ── embeddings ───────────────────────────────────────────────────────────────
class EmbeddingsCfg(_Section):
    model: str = "ViT-B-32"
    pretrained: str = "openai"
    batch_size: int = 16
    model_id: str = "google/siglip2-large-patch16-256"
    filename: str = "keyframe_embeddings.npy"
    mapping_filename: str = "index_mapping.json"
    min_similarity: float = 0.0


# ── audio_embeddings ─────────────────────────────────────────────────────────
class AudioEmbeddingsCfg(_Section):
    model_id: str = "laion/larger_clap_general"
    batch_size: int = 8
    chunk_seconds: float = 10.0
    sample_rate: int = 48000
    filename: str = "clap_embeddings.npy"
    mapping_filename: str = "audio_mapping.json"


# ── transcriber ──────────────────────────────────────────────────────────────
class TranscriberCfg(_Section):
    model_id: str = "Systran/faster-whisper-medium"
    compute_type: str = "auto"
    language: str | None = None
    beam_size: int = 5
    vad_filter: bool = True
    vad_min_silence_duration_ms: int = 500


# ── search (+ nested bm25) ───────────────────────────────────────────────────
class Bm25Cfg(_Section):
    k1: float = 1.5
    b: float = 0.75
    stopwords_lang: str | None = None
    rrf_k: int = 60
    include_transcripts: bool = True
    tokenizer: str = "regex"  # "regex" (default, unchanged) | "multilingual" (PT-aware, opt-in)


class SearchCfg(_Section):
    top_k_default: int = 9
    hybrid_sem_w: float = 0.70
    hybrid_bm25_w: float = 0.30
    hybrid_enabled: bool = True
    bm25: Bm25Cfg
    rerank_enabled: bool = False
    mmr_lambda: float = 0.5
    image_enabled: bool = True
    audio_enabled: bool = True
    multimodal_enabled: bool = True
    signals_enabled: bool = False


# ── retrieval (+ nested reranker/fusion/rhymes) ──────────────────────────────
class RerankerCfg(_Section):
    enabled: bool = False
    top_k_in: int = 20
    model: str = "default"


class FusionCfg(_Section):
    visual_weight: float = 0.5
    k_each: int = 50


class RhymesRetrievalCfg(_Section):
    diversity: float = 0.5
    k_candidates: int = 30
    k_final: int = 10


class RetrievalCfg(_Section):
    reranker: RerankerCfg
    fusion: FusionCfg
    rhymes: RhymesRetrievalCfg


# ── rimas ────────────────────────────────────────────────────────────────────
class RimasCfg(_Section):
    top_n: int = 8
    mmr_lambda: float = 0.5
    threshold: float = 0.75


# ── collaboration ────────────────────────────────────────────────────────────
class CollaborationCfg(_Section):
    composer_enabled: bool = False
    threads_enabled: bool = False
    demo_threads_enabled: bool = True


# ── llm ──────────────────────────────────────────────────────────────────────
class LlmCfg(_Section):
    model_id: str = "vikhyatk/moondream2"
    revision: str = "2025-01-09"
    checkpoint_interval: int = 25
    process_limit: int | None = None
    gpu_layers: int = -1
    descriptions_filename: str = "scene_descriptions.json"
    tags_filename: str = "scene_tags.json"


# ── models (selectors) ───────────────────────────────────────────────────────
class ModelsCfg(_Section):
    image_embedder: Literal["clip_openclip", "clip_mclip", "siglip_multilingual"]
    face_detector: Literal["mtcnn_pytorch"]
    object_detector: Literal["yolov8"]
    scene_describer: Literal["moondream_transformers", "moondream_gguf"]
    environment_classifier: Literal["opencv_heuristic"]
    audio_embedder: Literal["clap_hf"]
    transcriber: Literal["faster_whisper_hf"]


# ── pipeline (+ nested steps) ────────────────────────────────────────────────
class PipelineStepsCfg(_Section):
    frame_extraction: bool = True
    scene_detection: bool = True
    visual_analysis: bool = True
    embeddings: bool = True
    llm_description: bool = True
    audio_extract: bool = False
    audio_transcribe: bool = False
    audio_embed: bool = False


class PipelineCfg(_Section):
    steps: PipelineStepsCfg
    skip_existing: bool = True
    stop_on_error: bool = False


# ── proc ─────────────────────────────────────────────────────────────────────
class ProcCfg(_Section):
    gpu_metrics_enabled: bool = True


# ── eval ─────────────────────────────────────────────────────────────────────
class EvalCfg(_Section):
    root: str = "data/eval"
    run_id: str = "default"


# ── logging ──────────────────────────────────────────────────────────────────
class LoggingCfg(_Section):
    level: str = "INFO"
    to_file: bool = True
    filename: str = "cinemateca.log"
    json_logs: bool = False  # renamed from `json` to avoid Pydantic v2 shadow warning


# ── root ─────────────────────────────────────────────────────────────────────
class Settings(_Section):
    """Root typed configuration. ``Config`` aliases this.

    Built by :func:`cinemateca.config.load_config` from the merged YAML.
    Provides ``.to_dict()`` so :func:`cinemateca.run_manifest.config_snapshot`
    (which does ``hasattr(cfg, 'to_dict')``) keeps producing a stable,
    JSON-coercible snapshot byte-for-byte equivalent to the old
    ``_Namespace.to_dict()``.
    """

    config_version: int = CONFIG_VERSION
    project: ProjectCfg
    domain: DomainCfg
    paths: PathsCfg
    hardware: HardwareCfg
    frame_extraction: FrameExtractionCfg
    scene_detection: SceneDetectionCfg
    visual_analysis: VisualAnalysisCfg
    embeddings: EmbeddingsCfg
    audio_embeddings: AudioEmbeddingsCfg
    transcriber: TranscriberCfg
    search: SearchCfg
    retrieval: RetrievalCfg
    rimas: RimasCfg
    collaboration: CollaborationCfg
    llm: LlmCfg
    models: ModelsCfg
    pipeline: PipelineCfg
    proc: ProcCfg
    eval: EvalCfg
    logging: LoggingCfg

    seed: int = 42

    # ``to_dict`` is inherited from ``_Section`` — it returns a plain dict
    # with ``Path`` objects preserved, byte-for-byte equivalent to the old
    # ``_Namespace.to_dict()`` that ``run_manifest.config_snapshot`` reads.
