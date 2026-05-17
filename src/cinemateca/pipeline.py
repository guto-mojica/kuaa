"""
cinemateca.pipeline
~~~~~~~~~~~~~~~~~~~
Orquestrador do pipeline completo de catalogação audiovisual.

Executa as etapas na ordem correta, respeitando as configurações de
skip_existing e stop_on_error definidas na config.

Uso via código:
    from cinemateca.config import load_config
    from cinemateca.pipeline import CatalogPipeline

    cfg = load_config("config/local.yaml")
    pipeline = CatalogPipeline(cfg)
    results = pipeline.run("data/raw/meu_filme.mp4")

Uso via CLI:
    python -m cinemateca process --video data/raw/meu_filme.mp4
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Canonical pipeline step order. The single source of truth for which
# steps exist and in what order they run.
STEP_ORDER: tuple[str, ...] = (
    "frame_extraction",
    "scene_detection",
    "visual_analysis",
    "embeddings",
    "llm_description",
)

# Dependency graph for step execution.
#
# Each step lists the OTHER steps whose output it consumes. The edges
# below were verified against the private ``_step_*`` implementations and
# the legacy ``run()`` orchestrator (which is preserved verbatim):
#
#   * ``frame_extraction``  — root. Reads the video file only.
#   * ``scene_detection``   — root. ``_step_scene_detection`` reads the
#     video directly (it does NOT consume sampled frames); ``run()``
#     never gated it on frame_extraction. So it has no step prereq.
#   * ``visual_analysis``   — needs the keyframe ``.jpg`` files produced
#     by ``scene_detection`` (``_step_visual_analysis`` raises
#     ``FileNotFoundError`` when ``keyframes_dir`` is empty).
#   * ``embeddings``        — needs ``keyframes_metadata.json`` produced
#     by ``scene_detection`` (``run()`` gates it on
#     ``metadata_path.exists()``).
#   * ``llm_description``   — same metadata prerequisite as embeddings.
#
# Edges encode the *producing* step. The actual gate (see
# :meth:`CatalogPipeline.run_steps`) is INPUT-based: a downstream step is
# only blocked when its required artefact is genuinely absent AND not
# (re)produced by a prerequisite running in the same invocation. This
# preserves legitimate subset runs that rely on artefacts from a prior
# successful run.
STEP_DEPS: dict[str, tuple[str, ...]] = {
    "frame_extraction": (),
    "scene_detection": (),
    "visual_analysis": ("scene_detection",),
    "embeddings": ("scene_detection",),
    "llm_description": ("scene_detection",),
}


class StepCancelled(Exception):
    """Raised internally when a cooperative cancel is requested mid-run."""


@dataclass
class StepRun:
    """Per-step record produced by :meth:`CatalogPipeline.run_steps`.

    ``state`` is one of ``done`` / ``skipped`` / ``error`` / ``blocked``.
    ``blocked`` means a prerequisite failed or a required input artefact
    was missing, so the step was deliberately NOT executed (no stale
    mixed output is produced).
    """

    name: str
    state: str
    duration_s: float = 0.0
    error: str | None = None
    output: Any = None


@dataclass
class StepResults:
    """Aggregate result of a selected-step run."""

    video_path: str
    runs: list[StepRun] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(r.state in ("done", "skipped") for r in self.runs)


@dataclass
class StepResult:
    """Resultado de uma etapa do pipeline."""
    name: str
    success: bool
    skipped: bool = False
    duration_s: float = 0.0
    output: Any = None
    error: str | None = None


@dataclass
class PipelineResult:
    """Resultado completo de uma execução do pipeline."""
    video_path: str
    steps: list[StepResult] = field(default_factory=list)
    total_duration_s: float = 0.0

    @property
    def success(self) -> bool:
        return all(s.success or s.skipped for s in self.steps)

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"PIPELINE — {Path(self.video_path).name}",
            f"{'='*60}",
        ]
        for step in self.steps:
            if step.skipped:
                status = "⏭  PULADO"
            elif step.success:
                status = f"✓  OK ({step.duration_s:.1f}s)"
            else:
                status = "✗  ERRO"
            lines.append(f"  {step.name:<25} {status}")
            if step.error:
                lines.append(f"    → {step.error}")
        lines.append(f"{'='*60}")
        lines.append(f"  Total: {self.total_duration_s:.1f}s")
        lines.append(f"  Status: {'✓ OK' if self.success else '✗ COM ERROS'}")
        return "\n".join(lines)


class CatalogPipeline:
    """
    Executa o pipeline completo de catalogação audiovisual.

    Etapas:
        1. frame_extraction   — Extração de frames via FFmpeg
        2. scene_detection    — Detecção de cortes e extração de keyframes
        3. visual_analysis    — MTCNN + YOLO + classificação de ambiente
        4. embeddings         — Geração de embeddings CLIP
        5. llm_description    — Geração de metadados com Moondream 2

    Cada etapa pode ser pulada (skip_existing=True ou step desativado na config).
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self._device = None
        self._embedder = None
        self._describer = None

    @property
    def device(self):
        if self._device is None:
            from cinemateca.device import device_from_config
            self._device = device_from_config(self.cfg)
        return self._device

    # ─── Etapas individuais ───────────────────────────────────────────────────

    def _step_frame_extraction(self, video_path: Path) -> StepResult:
        from cinemateca.data_prep import FrameExtractor, VideoInspector

        name = "frame_extraction"
        output_dir = self.cfg.paths.frames_dir / "sample"

        # Skip se já existirem frames
        existing = sorted(output_dir.glob("*.jpg"))
        if self.cfg.pipeline.skip_existing and existing:
            logger.info("↷ Pulando frame_extraction (%d frames existentes)", len(existing))
            return StepResult(name=name, success=True, skipped=True, output=existing)

        t0 = time.time()
        try:
            inspector = VideoInspector(video_path)
            inspector.save_metadata(
                self.cfg.paths.metadata_dir / "video_properties.json"
            )
            extractor = FrameExtractor(self.cfg)
            frames = extractor.extract(video_path, output_dir)
            return StepResult(
                name=name, success=True, duration_s=time.time() - t0, output=frames
            )
        except Exception as e:
            return StepResult(name=name, success=False, duration_s=time.time() - t0, error=str(e))

    def _step_scene_detection(self, video_path: Path) -> StepResult:
        from cinemateca.scene_detector import SceneDetector

        name = "scene_detection"
        metadata_path = self.cfg.paths.metadata_dir / "keyframes_metadata.json"
        keyframes_dir = self.cfg.paths.frames_dir / "scenes" / "keyframes_content"

        if self.cfg.pipeline.skip_existing and metadata_path.exists():
            logger.info("↷ Pulando scene_detection (metadados existentes)")
            return StepResult(
                name=name, success=True, skipped=True,
                output={"metadata_path": metadata_path, "keyframes_dir": keyframes_dir}
            )

        t0 = time.time()
        try:
            detector = SceneDetector(self.cfg)
            scene_list = detector.detect(video_path)
            keyframes = detector.extract_keyframes(scene_list, video_path, keyframes_dir)
            metadata_path = detector.export_metadata(scene_list, keyframes, metadata_path)
            stats = detector.scene_stats(scene_list)
            logger.info(
                "Cenas: %d detectadas | média %.1fs | keyframes: %d",
                stats.get("num_scenes", 0),
                stats.get("mean_s", 0),
                len(keyframes),
            )
            return StepResult(
                name=name, success=True, duration_s=time.time() - t0,
                output={"metadata_path": metadata_path, "keyframes": keyframes, "stats": stats}
            )
        except Exception as e:
            return StepResult(name=name, success=False, duration_s=time.time() - t0, error=str(e))

    def _step_visual_analysis(self, keyframes_dir: Path) -> StepResult:
        from cinemateca.models.environment.opencv_heuristic import OpenCVEnvironmentClassifier
        from cinemateca.models.face.mtcnn import MTCNNFaceDetector
        from cinemateca.models.objects.yolov8 import YOLOv8ObjectDetector
        from cinemateca.visual_analyzer import VisualAnalyzer

        name = "visual_analysis"
        output_path = self.cfg.paths.metadata_dir / "visual_analysis.json"

        if self.cfg.pipeline.skip_existing and output_path.exists():
            logger.info("↷ Pulando visual_analysis (metadados existentes)")
            return StepResult(name=name, success=True, skipped=True, output=output_path)

        t0 = time.time()
        try:
            keyframes = sorted(keyframes_dir.glob("*.jpg"))
            if not keyframes:
                raise FileNotFoundError(f"Nenhum keyframe em: {keyframes_dir}")

            analyzer = VisualAnalyzer(
                face_detector=MTCNNFaceDetector(self.cfg, self.device),
                object_detector=YOLOv8ObjectDetector(self.cfg, self.device),
                env_classifier=OpenCVEnvironmentClassifier(self.cfg),
            )
            results = analyzer.analyze_keyframes(keyframes)
            analyzer.save_metadata(results, output_path)
            return StepResult(
                name=name, success=True, duration_s=time.time() - t0, output=output_path
            )
        except Exception as e:
            return StepResult(name=name, success=False, duration_s=time.time() - t0, error=str(e))

    def _step_embeddings(self, metadata_path: Path) -> StepResult:
        import pandas as pd

        from cinemateca.embeddings import CLIPEmbedder

        name = "embeddings"
        emb_cfg = self.cfg.embeddings
        emb_path = self.cfg.paths.embeddings_dir / emb_cfg.filename
        map_path = self.cfg.paths.embeddings_dir / emb_cfg.mapping_filename

        if self.cfg.pipeline.skip_existing and emb_path.exists():
            logger.info("↷ Pulando embeddings (arquivo existente)")
            return StepResult(
                name=name, success=True, skipped=True,
                output={"embeddings_path": emb_path, "mapping_path": map_path}
            )

        t0 = time.time()
        try:
            with open(metadata_path, encoding="utf-8") as f:
                kf_data = json.load(f)
            kf_df = pd.DataFrame(kf_data)
            kf_df["exists"] = kf_df["filepath"].apply(lambda x: Path(x).exists())
            valid_kf = kf_df[kf_df["exists"]].reset_index(drop=True)

            embedder = CLIPEmbedder(self.cfg, self.device)
            self._embedder = embedder  # reutilizar no step seguinte se necessário

            image_paths = [Path(p) for p in valid_kf["filepath"]]
            embeddings = embedder.encode_images(image_paths)
            emb_path, map_path = embedder.save(
                embeddings, valid_kf,
                self.cfg.paths.embeddings_dir,
                emb_cfg.filename,
                emb_cfg.mapping_filename,
            )
            return StepResult(
                name=name, success=True, duration_s=time.time() - t0,
                output={"embeddings_path": emb_path, "mapping_path": map_path}
            )
        except Exception as e:
            return StepResult(name=name, success=False, duration_s=time.time() - t0, error=str(e))

    def _step_llm_description(self, metadata_path: Path) -> StepResult:
        import pandas as pd

        from cinemateca.llm_describer import LLMDescriber

        name = "llm_description"
        llm_cfg = self.cfg.llm
        desc_path = self.cfg.paths.metadata_dir / llm_cfg.descriptions_filename
        tags_path = self.cfg.paths.metadata_dir / llm_cfg.tags_filename

        t0 = time.time()
        try:
            with open(metadata_path, encoding="utf-8") as f:
                kf_data = json.load(f)
            kf_df = pd.DataFrame(kf_data)
            kf_df["exists"] = kf_df["filepath"].apply(lambda x: Path(x).exists())
            valid_kf = kf_df[kf_df["exists"]].reset_index(drop=True)

            # Pular apenas se TODAS as cenas válidas já foram descritas
            if self.cfg.pipeline.skip_existing and desc_path.exists():
                with open(desc_path, encoding="utf-8") as f:
                    existing_check = json.load(f)
                described_ids = {r["scene_id"] for r in existing_check if "error" not in r}
                pending = len(valid_kf[~valid_kf["scene_id"].isin(described_ids)])
                if pending == 0:
                    logger.info("↷ Pulando llm_description (todas as %d cenas já descritas)", len(described_ids))
                    return StepResult(
                        name=name, success=True, skipped=True,
                        output={"descriptions_path": desc_path, "tags_path": tags_path}
                    )
                logger.info("↷ llm_description: %d cenas pendentes, retomando...", pending)

            # Retomada: carregar resultados anteriores se existirem
            existing = []
            if desc_path.exists():
                with open(desc_path, encoding="utf-8") as f:
                    existing = json.load(f)
                logger.info("Retomando: %d cenas já descritas", len(existing))

            describer = LLMDescriber(self.cfg, self.device)
            results = describer.describe_keyframes(
                valid_kf,
                existing_results=existing,
                checkpoint_path=desc_path,
            )
            tag_index = describer.build_tag_index(results)
            describer.save(results, tag_index, self.cfg.paths.metadata_dir)

            return StepResult(
                name=name, success=True, duration_s=time.time() - t0,
                output={"descriptions_path": desc_path, "tags_path": tags_path}
            )
        except Exception as e:
            return StepResult(name=name, success=False, duration_s=time.time() - t0, error=str(e))

    # ─── Orquestrador principal ───────────────────────────────────────────────

    def run(self, video_path: str | Path) -> PipelineResult:
        """
        Executa o pipeline completo para um arquivo de vídeo.

        Args:
            video_path: Caminho do arquivo de vídeo a processar.

        Returns:
            PipelineResult com o status de cada etapa.
        """
        video_path = Path(video_path)
        pipeline_start = time.time()

        logger.info("=" * 60)
        logger.info("PIPELINE INICIADO: %s", video_path.name)
        logger.info("=" * 60)

        result = PipelineResult(video_path=str(video_path))
        steps_cfg = self.cfg.pipeline.steps
        keyframes_dir = self.cfg.paths.frames_dir / "scenes" / "keyframes_content"
        metadata_path = self.cfg.paths.metadata_dir / "keyframes_metadata.json"

        # ── Etapa 1: Extração de frames ───────────────────────────────────────
        if steps_cfg.frame_extraction:
            step = self._step_frame_extraction(video_path)
            result.steps.append(step)
            if not step.success and self.cfg.pipeline.stop_on_error:
                logger.error("Pipeline interrompido na etapa: %s", step.name)
                result.total_duration_s = time.time() - pipeline_start
                return result
        else:
            result.steps.append(StepResult(name="frame_extraction", success=True, skipped=True))

        # ── Etapa 2: Detecção de cenas ────────────────────────────────────────
        if steps_cfg.scene_detection:
            step = self._step_scene_detection(video_path)
            result.steps.append(step)
            if step.success and not step.skipped:
                keyframes_dir = Path(
                    step.output.get("keyframes_dir", keyframes_dir)
                ) if isinstance(step.output, dict) else keyframes_dir
            if not step.success and self.cfg.pipeline.stop_on_error:
                logger.error("Pipeline interrompido na etapa: %s", step.name)
                result.total_duration_s = time.time() - pipeline_start
                return result
        else:
            result.steps.append(StepResult(name="scene_detection", success=True, skipped=True))

        # ── Etapa 3: Análise visual ───────────────────────────────────────────
        if steps_cfg.visual_analysis:
            step = self._step_visual_analysis(keyframes_dir)
            result.steps.append(step)
            if not step.success and self.cfg.pipeline.stop_on_error:
                logger.error("Pipeline interrompido na etapa: %s", step.name)
                result.total_duration_s = time.time() - pipeline_start
                return result
        else:
            result.steps.append(StepResult(name="visual_analysis", success=True, skipped=True))

        # ── Etapa 4: Embeddings ───────────────────────────────────────────────
        if steps_cfg.embeddings and metadata_path.exists():
            step = self._step_embeddings(metadata_path)
            result.steps.append(step)
            if not step.success and self.cfg.pipeline.stop_on_error:
                logger.error("Pipeline interrompido na etapa: %s", step.name)
                result.total_duration_s = time.time() - pipeline_start
                return result
        else:
            result.steps.append(StepResult(name="embeddings", success=True, skipped=True))

        # ── Etapa 5: Descrição LLM ────────────────────────────────────────────
        if steps_cfg.llm_description and metadata_path.exists():
            step = self._step_llm_description(metadata_path)
            result.steps.append(step)
        else:
            result.steps.append(StepResult(name="llm_description", success=True, skipped=True))

        result.total_duration_s = time.time() - pipeline_start
        logger.info(result.summary())
        return result

    # ─── Public selected-step API (Phase 4) ───────────────────────────────────

    def _keyframes_dir(self) -> Path:
        return self.cfg.paths.frames_dir / "scenes" / "keyframes_content"

    def _keyframes_metadata_path(self) -> Path:
        return self.cfg.paths.metadata_dir / "keyframes_metadata.json"

    def _inputs_available(self, step: str, keyframes_dir: Path) -> bool:
        """Return True if ``step``'s required input artefacts exist on disk.

        This is the INPUT-based gate: it lets a subset run proceed when
        the artefacts a prior successful run produced are still present,
        and only reports missing inputs when they are genuinely absent.
        """
        if step in ("frame_extraction", "scene_detection"):
            return True
        if step == "visual_analysis":
            return bool(sorted(keyframes_dir.glob("*.jpg")))
        if step in ("embeddings", "llm_description"):
            return self._keyframes_metadata_path().exists()
        return True

    def run_steps(
        self,
        video_path: str | Path,
        steps: list[str],
        progress_cb: Callable[[str, str, StepRun | None], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> StepResults:
        """Execute the requested ``steps`` for ``video_path``.

        Thin orchestration over the existing private ``_step_*``
        implementations — step logic is NOT reimplemented here. Adds:

          * step selection (only steps in ``steps`` run; others are not
            reported),
          * dependency-aware gating: a step is marked ``blocked`` (and
            NOT executed) when a prerequisite step running in this same
            invocation failed/blocked, or when its required input
            artefacts are absent on disk — preventing the historical
            defect where a failed ``scene_detection`` still let
            ``embeddings``/``llm_description`` run on a stale
            ``keyframes_metadata.json``,
          * per-step progress callback ``progress_cb(step_name, phase,
            run)`` where ``phase`` is ``"start"`` (run is ``None``) or
            ``"finish"`` (run is the completed :class:`StepRun`),
          * cooperative cancellation: ``cancel_check`` is polled before
            each step; when it returns truthy a :class:`StepCancelled`
            is raised so the caller can finalize a ``cancelled`` job.

        Successful full-run behaviour is unchanged: the same ``_step_*``
        methods run in the same order with the same skip-existing logic,
        producing byte-identical artefacts. Gating only changes the
        FAILURE / missing-input path.

        Args:
            video_path: Source video file.
            steps: Step names to execute (any subset of
                :data:`STEP_ORDER`).
            progress_cb: Optional per-step lifecycle callback.
            cancel_check: Optional callable polled between steps; truthy
                aborts the run via :class:`StepCancelled`.

        Returns:
            :class:`StepResults` with one :class:`StepRun` per requested
            step.

        Raises:
            StepCancelled: if ``cancel_check`` requested cancellation.
        """
        video_path = Path(video_path)
        requested = [s for s in STEP_ORDER if s in set(steps)]
        results = StepResults(video_path=str(video_path))

        keyframes_dir = self._keyframes_dir()
        metadata_path = self._keyframes_metadata_path()

        # Track per-step outcome for in-run dependency gating.
        outcome: dict[str, str] = {}

        def _emit(name: str, phase: str, run: StepRun | None) -> None:
            if progress_cb is not None:
                progress_cb(name, phase, run)

        for name in requested:
            if cancel_check is not None and cancel_check():
                raise StepCancelled()

            # ── Dependency gate ───────────────────────────────────────
            # A step is blocked if a prerequisite that ran in THIS
            # invocation did not succeed, OR its required inputs are not
            # on disk (and no in-run prerequisite will produce them).
            blocked_reason: str | None = None
            for dep in STEP_DEPS[name]:
                if dep in outcome and outcome[dep] not in ("done", "skipped"):
                    blocked_reason = (
                        f"prerequisite '{dep}' did not succeed "
                        f"(state: {outcome[dep]})"
                    )
                    break
            if blocked_reason is None and not self._inputs_available(
                name, keyframes_dir
            ):
                blocked_reason = (
                    f"required input artefacts for '{name}' are missing"
                )

            if blocked_reason is not None:
                run = StepRun(
                    name=name, state="blocked", error=blocked_reason
                )
                outcome[name] = "blocked"
                results.runs.append(run)
                _emit(name, "start", None)
                _emit(name, "finish", run)
                logger.warning(
                    "Step %s blocked: %s", name, blocked_reason
                )
                continue

            _emit(name, "start", None)

            # ── Delegate to the existing private step impl ────────────
            if name == "frame_extraction":
                sr = self._step_frame_extraction(video_path)
            elif name == "scene_detection":
                sr = self._step_scene_detection(video_path)
                if (
                    sr.success
                    and not sr.skipped
                    and isinstance(sr.output, dict)
                    and sr.output.get("keyframes_dir")
                ):
                    keyframes_dir = Path(sr.output["keyframes_dir"])
            elif name == "visual_analysis":
                sr = self._step_visual_analysis(keyframes_dir)
            elif name == "embeddings":
                sr = self._step_embeddings(metadata_path)
            elif name == "llm_description":
                sr = self._step_llm_description(metadata_path)
            else:  # pragma: no cover - guarded by requested filter
                raise ValueError(f"Unknown step: {name}")

            if sr.skipped:
                state = "skipped"
            elif sr.success:
                state = "done"
            else:
                state = "error"
            run = StepRun(
                name=name,
                state=state,
                duration_s=sr.duration_s,
                error=sr.error,
                output=sr.output,
            )
            outcome[name] = state
            results.runs.append(run)
            _emit(name, "finish", run)

        return results
