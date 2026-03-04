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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    """Resultado de uma etapa do pipeline."""
    name: str
    success: bool
    skipped: bool = False
    duration_s: float = 0.0
    output: Any = None
    error: Optional[str] = None


@dataclass
class PipelineResult:
    """Resultado completo de uma execução do pipeline."""
    video_path: str
    steps: List[StepResult] = field(default_factory=list)
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
                status = f"✗  ERRO"
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
        from cinemateca.data_prep import VideoInspector, FrameExtractor

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

            analyzer = VisualAnalyzer(self.cfg, self.device)
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

        if self.cfg.pipeline.skip_existing and desc_path.exists():
            logger.info("↷ Pulando llm_description (descrições existentes)")
            return StepResult(
                name=name, success=True, skipped=True,
                output={"descriptions_path": desc_path, "tags_path": tags_path}
            )

        t0 = time.time()
        try:
            with open(metadata_path, encoding="utf-8") as f:
                kf_data = json.load(f)
            kf_df = pd.DataFrame(kf_data)
            kf_df["exists"] = kf_df["filepath"].apply(lambda x: Path(x).exists())
            valid_kf = kf_df[kf_df["exists"]].reset_index(drop=True)

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
