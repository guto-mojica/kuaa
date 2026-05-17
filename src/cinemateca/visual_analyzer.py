"""
cinemateca.visual_analyzer
~~~~~~~~~~~~~~~~~~~~~~~~~~
VisualAnalyzer facade. Composes injected Face / Object / Environment
backends (provided by cinemateca.models.registry). The detector classes
themselves live under cinemateca.models.{face,objects,environment}.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class VisualAnalyzer:
    """Orchestrates injected face / object / environment backends.

    Backends are supplied by the caller (the pipeline builds them via
    cinemateca.models.registry). Saves one consolidated visual-metadata
    JSON, format-compatible with the embeddings and LLM modules.
    """

    def __init__(self, face_detector, object_detector, env_classifier):
        self.face_detector = face_detector
        self.object_detector = object_detector
        self.env_classifier = env_classifier

    def analyze_frame(self, image_path: str | Path) -> dict:
        path = Path(image_path)
        return {
            "frame_path": path.name,
            "face_detection": self.face_detector.detect(path),
            "object_detection": self.object_detector.detect(path),
            "environment": self.env_classifier.classify(path),
        }

    def analyze_keyframes(
        self,
        keyframe_paths: list[Path],
        max_frames: int | None = None,
    ) -> list[dict]:
        paths = keyframe_paths[:max_frames] if max_frames else keyframe_paths
        results = []
        for i, p in enumerate(paths):
            try:
                r = self.analyze_frame(p)
                results.append(r)
                if (i + 1) % 25 == 0:
                    logger.info("Análise visual: %d/%d frames", i + 1, len(paths))
            except Exception as e:
                logger.error("Erro ao analisar %s: %s", Path(p).name, e)
                results.append({"frame_path": Path(p).name, "error": str(e)})

        logger.info("✓ Análise visual concluída: %d frames", len(results))
        return results

    def save_metadata(self, results: list[dict], output_path: str | Path) -> Path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("✓ Metadados visuais salvos: %s (%d frames)", out, len(results))
        return out
