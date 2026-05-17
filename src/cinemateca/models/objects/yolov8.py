"""YOLOv8 object-detection backend (moved from visual_analyzer.py, unchanged)."""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class YOLOv8ObjectDetector:
    """Detects objects using YOLOv8 (Ultralytics)."""

    def __init__(self, cfg=None, device=None):
        self._model = None
        self._device = device

        if cfg is not None:
            od_cfg = cfg.visual_analysis.object_detection
            self.enabled = od_cfg.enabled
            self.model_name = od_cfg.model
            self.confidence = od_cfg.confidence
        else:
            self.enabled = True
            self.model_name = "yolov8n.pt"
            self.confidence = 0.30

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO
        except ImportError:
            raise RuntimeError(
                "ultralytics não instalado. Execute: pip install ultralytics"
            )
        self._model = YOLO(self.model_name)
        logger.info("YOLOv8 carregado: %s", self.model_name)

    def detect(self, image_path: str | Path) -> dict:
        if not self.enabled:
            return {"num_objects": 0, "objects": [], "class_counts": {}}

        self._load_model()
        results = self._model(str(image_path), conf=self.confidence, verbose=False)

        objects = []
        class_counts: dict[str, int] = {}

        for result in results:
            for box in result.boxes:
                cls_name = self._model.names[int(box.cls[0])]
                obj = {
                    "class": cls_name,
                    "class_id": int(box.cls[0]),
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].tolist(),
                }
                objects.append(obj)
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

        return {
            "num_objects": len(objects),
            "objects": objects,
            "class_counts": class_counts,
        }

    def detect_batch(self, image_paths: list[Path]) -> list[dict]:
        results = []
        for p in image_paths:
            r = self.detect(p)
            r["frame_path"] = str(p.name)
            results.append(r)
        logger.info("Detecção de objetos: %d frames processados", len(results))
        return results
