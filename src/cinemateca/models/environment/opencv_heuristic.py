"""OpenCV-heuristic environment classifier (moved, unchanged)."""
from __future__ import annotations

import logging
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)


class OpenCVEnvironmentClassifier:
    """Classifies scene environment via brightness + edge-density heuristics."""

    def __init__(self, cfg=None):
        if cfg is not None:
            env_cfg = cfg.visual_analysis.environment
            self.enabled = env_cfg.enabled
            self.brightness_threshold = env_cfg.brightness_threshold
            self.edge_density_threshold = env_cfg.edge_density_threshold
        else:
            self.enabled = True
            self.brightness_threshold = 100
            self.edge_density_threshold = 0.05

    def classify(self, image_path: str | Path) -> dict:
        if not self.enabled:
            return {
                "time_of_day": "desconhecido",
                "brightness_score": 0.0,
                "location": "desconhecido",
                "edge_density": 0.0,
            }

        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning("Não foi possível ler frame: %s", image_path)
            return {
                "time_of_day": "desconhecido",
                "brightness_score": 0.0,
                "location": "desconhecido",
                "edge_density": 0.0,
            }

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())

        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(edges.sum() / (edges.shape[0] * edges.shape[1]))

        return {
            "time_of_day": "dia" if brightness > self.brightness_threshold else "noite",
            "brightness_score": brightness,
            "location": "exterior" if edge_density > self.edge_density_threshold else "interior",
            "edge_density": edge_density,
        }

    def classify_batch(self, image_paths: list[Path]) -> list[dict]:
        results = []
        for p in image_paths:
            r = self.classify(p)
            r["frame_path"] = str(p.name)
            results.append(r)
        logger.info("Classificação de ambiente: %d frames processados", len(results))
        return results
