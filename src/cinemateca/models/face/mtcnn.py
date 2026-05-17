"""MTCNN face-detection backend (moved from visual_analyzer.py, unchanged)."""
from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)


class MTCNNFaceDetector:
    """Detects faces in frames using MTCNN (facenet-pytorch)."""

    def __init__(self, cfg=None, device=None):
        self._model = None
        self._device = device

        va_cfg = getattr(cfg, "visual_analysis", None) if cfg is not None else None
        fd_cfg = getattr(va_cfg, "face_detection", None) if va_cfg is not None else None
        if fd_cfg is not None:
            self.enabled = fd_cfg.enabled
            self.min_face_size = fd_cfg.min_face_size
            self.thresholds = list(fd_cfg.thresholds)
        else:
            self.enabled = True
            self.min_face_size = 20
            self.thresholds = [0.6, 0.7, 0.7]

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from facenet_pytorch import MTCNN
        except ImportError:
            raise RuntimeError(
                "facenet-pytorch não instalado. Execute: pip install facenet-pytorch"
            )

        # MTCNN has MPS incompatibilities with adaptive pooling — always use CPU
        self._model = MTCNN(
            keep_all=True,
            device="cpu",
            thresholds=self.thresholds,
            min_face_size=self.min_face_size,
        )
        logger.info("MTCNN carregado no device: cpu")

    def detect(self, image_path: str | Path) -> dict:
        if not self.enabled:
            return {"num_faces": 0, "faces": []}

        self._load_model()

        img = Image.open(image_path)
        boxes, probs, landmarks = self._model.detect(img, landmarks=True)

        if boxes is None:
            return {"num_faces": 0, "faces": []}

        faces = []
        for box, prob, lm in zip(boxes, probs, landmarks):
            faces.append({
                "bbox": box.tolist(),
                "confidence": float(prob),
                "landmarks": lm.tolist() if lm is not None else None,
                "area": float((box[2] - box[0]) * (box[3] - box[1])),
            })

        return {"num_faces": len(faces), "faces": faces}

    def detect_batch(self, image_paths: list[Path]) -> list[dict]:
        results = []
        for p in image_paths:
            r = self.detect(p)
            r["frame_path"] = str(p.name)
            results.append(r)
        logger.info("Detecção facial: %d frames processados", len(results))
        return results
