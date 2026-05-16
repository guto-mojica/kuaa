"""
cinemateca.scene_detector
~~~~~~~~~~~~~~~~~~~~~~~~~
Detecção de cortes de cena e extração de keyframes usando PySceneDetect.

Baseado no Notebook 02 (02_deteccao_cenas.ipynb).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# PySceneDetect — importação diferida para não quebrar se não instalado
try:
    from scenedetect import SceneManager, open_video
    from scenedetect.detectors import AdaptiveDetector, ContentDetector
    from scenedetect.scene_manager import save_images
    _SCENEDETECT_AVAILABLE = True
except ImportError:
    _SCENEDETECT_AVAILABLE = False
    logger.warning("PySceneDetect não instalado. Instale com: pip install scenedetect[opencv]")


SceneList = list[tuple]   # list of (FrameTimecode, FrameTimecode)


class SceneDetector:
    """
    Detecta cenas em um arquivo de vídeo e extrai keyframes representativos.

    Suporta dois algoritmos:
        "content"  — ContentDetector: diferença de histograma entre frames.
                     Bom para cortes diretos (hard cuts).
        "adaptive" — AdaptiveDetector: threshold adaptativo baseado em
                     variações locais. Melhor para dissolvências (fades).

    Exemplo:
        detector = SceneDetector(cfg)
        scenes = detector.detect("data/raw/jeca_tatu.mp4")
        keyframes = detector.extract_keyframes(scenes, "data/raw/jeca_tatu.mp4",
                                               "data/frames/scenes")
    """

    def __init__(self, cfg=None):
        if cfg is not None:
            sd = cfg.scene_detection
            self.detector_type = sd.detector
            self.content_threshold = sd.content_threshold
            self.adaptive_threshold = sd.adaptive_threshold
            self.min_scene_len = sd.min_scene_len
            self.keyframes_per_scene = sd.keyframes_per_scene
            self.keyframe_height = getattr(sd, "keyframe_height", 480)
        else:
            self.detector_type = "adaptive"
            self.content_threshold = 27.0
            self.adaptive_threshold = 3.0
            self.min_scene_len = 15
            self.keyframes_per_scene = 3
            self.keyframe_height = 480

    def detect(self, video_path: str | Path) -> SceneList:
        """
        Detecta cenas no vídeo.

        Args:
            video_path: Caminho do arquivo de vídeo.

        Returns:
            Lista de tuplas (start_timecode, end_timecode).

        Raises:
            RuntimeError: Se PySceneDetect não estiver instalado.
            FileNotFoundError: Se o vídeo não existir.
        """
        if not _SCENEDETECT_AVAILABLE:
            raise RuntimeError(
                "PySceneDetect não instalado. "
                "Execute: pip install scenedetect[opencv]"
            )

        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")

        video_manager = open_video(str(video_path))
        scene_manager = SceneManager()

        if self.detector_type == "adaptive":
            scene_manager.add_detector(
                AdaptiveDetector(
                    adaptive_threshold=self.adaptive_threshold,
                    min_scene_len=self.min_scene_len,
                )
            )
            logger.info(
                "Detectando cenas com AdaptiveDetector (threshold=%.1f, min_len=%d)",
                self.adaptive_threshold,
                self.min_scene_len,
            )
        else:
            scene_manager.add_detector(
                ContentDetector(
                    threshold=self.content_threshold,
                    min_scene_len=self.min_scene_len,
                )
            )
            logger.info(
                "Detectando cenas com ContentDetector (threshold=%.1f, min_len=%d)",
                self.content_threshold,
                self.min_scene_len,
            )

        scene_manager.detect_scenes(video_manager, show_progress=True)
        scene_list = scene_manager.get_scene_list()
        video_manager.capture.release()

        logger.info("✓ %d cenas detectadas em %s", len(scene_list), video_path.name)
        return scene_list

    def extract_keyframes(
        self,
        scene_list: SceneList,
        video_path: str | Path,
        output_dir: str | Path,
    ) -> list[Path]:
        """
        Extrai keyframes representativos de cada cena.

        Args:
            scene_list: Saída de self.detect().
            video_path: Caminho do vídeo original.
            output_dir: Diretório onde os keyframes serão salvos.

        Returns:
            Lista ordenada dos Paths dos keyframes extraídos.
        """
        if not _SCENEDETECT_AVAILABLE:
            raise RuntimeError("PySceneDetect não instalado.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        video_manager = open_video(str(video_path))
        save_images(
            scene_list,
            video_manager,
            num_images=self.keyframes_per_scene,
            image_extension="jpg",
            output_dir=str(output_dir),
            height=self.keyframe_height if self.keyframe_height > 0 else None,
        )
        video_manager.capture.release()

        keyframes = sorted(output_dir.glob("*.jpg"))
        logger.info("✓ %d keyframes extraídos em %s", len(keyframes), output_dir)
        return keyframes

    def scene_stats(self, scene_list: SceneList) -> dict:
        """
        Calcula estatísticas descritivas sobre as durações das cenas.

        Args:
            scene_list: Saída de self.detect().

        Returns:
            dict com num_scenes, total/mean/median/min/max/std de duração em segundos.
        """
        import numpy as np

        if not scene_list:
            return {}

        durations = np.array([
            (end - start).get_seconds()
            for start, end in scene_list
        ])

        return {
            "num_scenes": len(scene_list),
            "total_duration_s": float(durations.sum()),
            "mean_s": float(durations.mean()),
            "median_s": float(np.median(durations)),
            "min_s": float(durations.min()),
            "max_s": float(durations.max()),
            "std_s": float(durations.std()),
        }

    def export_metadata(
        self,
        scene_list: SceneList,
        keyframe_paths: list[Path],
        output_path: str | Path,
    ) -> Path:
        """
        Exporta metadados das cenas para JSON.

        O JSON gerado inclui scene_id, timecodes e path do keyframe,
        compatível com o formato esperado pelos módulos seguintes.

        Args:
            scene_list:     Saída de self.detect().
            keyframe_paths: Saída de self.extract_keyframes().
            output_path:    Caminho do arquivo JSON de saída.

        Returns:
            Path do arquivo JSON criado.
        """
        # Mapear um keyframe por cena (alinha por índice, ajusta se sobrarem)
        # PySceneDetect pode gerar nomes como Scene-001-01.jpg
        kf_per_scene = max(1, self.keyframes_per_scene)
        scenes_data = []

        for idx, (start, end) in enumerate(scene_list):
            # Pegar o keyframe do meio desta cena
            kf_idx = idx * kf_per_scene + (kf_per_scene // 2)
            kf_path = str(keyframe_paths[kf_idx]) if kf_idx < len(keyframe_paths) else ""

            scenes_data.append({
                "scene_id": idx + 1,
                "keyframe_id": f"scene_{idx+1:04d}",
                "filepath": kf_path,
                "start_time_s": start.get_seconds(),
                "end_time_s": end.get_seconds(),
                "duration_s": (end - start).get_seconds(),
                "start_frame": start.get_frames(),
                "end_frame": end.get_frames(),
            })

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(scenes_data, f, indent=2, ensure_ascii=False)

        logger.info(
            "✓ Metadados de %d cenas exportados: %s",
            len(scenes_data),
            output_path,
        )
        return output_path
