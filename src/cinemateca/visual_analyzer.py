"""
cinemateca.visual_analyzer
~~~~~~~~~~~~~~~~~~~~~~~~~~
Análise visual de frames: detecção facial (MTCNN), detecção de objetos
(YOLOv8) e classificação de ambiente (heurística).

Baseado no Notebook 03 (03_analise_visual.ipynb).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
from PIL import Image

logger = logging.getLogger(__name__)


# ─── Detecção Facial ─────────────────────────────────────────────────────────

class FaceDetector:
    """
    Detecta rostos em frames usando MTCNN (Multi-task Cascaded CNN).

    MTCNN é robusto a variações de escala e pose, adequado para
    filmes com qualidade variável de digitalização.
    """

    def __init__(self, cfg=None, device=None):
        self._model = None
        self._device = device

        if cfg is not None:
            fd_cfg = cfg.visual_analysis.face_detection
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
        """
        Detecta rostos em uma imagem.

        Args:
            image_path: Caminho da imagem.

        Returns:
            dict com 'num_faces' (int) e 'faces' (list de dicts com bbox,
            confidence, landmarks e área).
        """
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


# ─── Detecção de Objetos ──────────────────────────────────────────────────────

class ObjectDetector:
    """
    Detecta objetos usando YOLOv8 (Ultralytics).

    O modelo 'nano' (yolov8n) é o default para velocidade.
    Para maior acurácia em produção, use 'yolov8s' ou 'yolov8m'.
    """

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
        """
        Detecta objetos em uma imagem.

        Args:
            image_path: Caminho da imagem.

        Returns:
            dict com 'num_objects', 'objects' (list) e 'class_counts' (dict).
        """
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


# ─── Classificação de Ambiente ────────────────────────────────────────────────

class EnvironmentClassifier:
    """
    Classifica o ambiente de uma cena usando heurísticas de visão computacional.

    AVISO: Esta é uma implementação aproximada baseada em métricas simples de
    brilho e densidade de bordas. Para uso em produção, recomenda-se treinar
    um classificador específico para o acervo da instituição.

    Classificações:
        time_of_day : "dia" | "noite"
        location    : "exterior" | "interior"
    """

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
        """
        Classifica o ambiente de um frame.

        Args:
            image_path: Caminho da imagem.

        Returns:
            dict com time_of_day, brightness_score, location, edge_density.
        """
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


# ─── Analisador Unificado ─────────────────────────────────────────────────────

class VisualAnalyzer:
    """
    Orquestra FaceDetector, ObjectDetector e EnvironmentClassifier.

    Salva um único JSON consolidado com todos os metadados visuais,
    compatível com o formato esperado pelos módulos de embeddings e LLM.

    Exemplo:
        analyzer = VisualAnalyzer(cfg, device)
        results = analyzer.analyze_keyframes(keyframe_paths)
        analyzer.save_metadata(results, cfg.paths.metadata_dir / "visual_analysis.json")
    """

    def __init__(self, cfg=None, device=None):
        self.face_detector = FaceDetector(cfg, device)
        self.object_detector = ObjectDetector(cfg, device)
        self.env_classifier = EnvironmentClassifier(cfg)

    def analyze_frame(self, image_path: str | Path) -> dict:
        """Analisa um único frame com todos os detectores."""
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
        """
        Analisa uma lista de keyframes.

        Args:
            keyframe_paths: Lista de Paths das imagens a analisar.
            max_frames:     Limitar análise aos primeiros N frames (para testes).

        Returns:
            Lista de dicts com metadados visuais consolidados por frame.
        """
        paths = keyframe_paths[:max_frames] if max_frames else keyframe_paths
        results = []
        for i, p in enumerate(paths):
            try:
                r = self.analyze_frame(p)
                results.append(r)
                if (i + 1) % 25 == 0:
                    logger.info("Análise visual: %d/%d frames", i + 1, len(paths))
            except Exception as e:
                logger.error("Erro ao analisar %s: %s", p.name, e)
                results.append({"frame_path": p.name, "error": str(e)})

        logger.info("✓ Análise visual concluída: %d frames", len(results))
        return results

    def save_metadata(self, results: list[dict], output_path: str | Path) -> Path:
        """Salva os resultados consolidados em JSON."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("✓ Metadados visuais salvos: %s (%d frames)", out, len(results))
        return out
