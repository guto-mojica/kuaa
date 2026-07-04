"""
kuaa.data_prep
~~~~~~~~~~~~~~~~~~~~
Preparação de dados: inspeção de vídeo (FFprobe) e análise de qualidade de frames.

Baseado no Notebook 01 (01_preparacao_dados.ipynb).
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoInspector:
    """
    Extrai propriedades técnicas de um arquivo de vídeo usando FFprobe.

    Exemplo:
        inspector = VideoInspector("data/raw/jeca_tatu.mp4")
        props = inspector.properties
        print(f"{props['duration_minutes']:.1f} minutos")
    """

    def __init__(self, video_path: str | Path):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Vídeo não encontrado: {self.video_path}")
        self._properties: dict | None = None

    @property
    def properties(self) -> dict:
        if self._properties is None:
            self._properties = self._probe()
        return self._properties

    def _probe(self) -> dict:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(self.video_path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFprobe falhou: {e.stderr}") from e
        except FileNotFoundError:
            raise RuntimeError(
                "FFprobe não encontrado. Instale o FFmpeg: https://ffmpeg.org/download.html"
            )

        data = json.loads(result.stdout)
        video_stream = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
        if not video_stream:
            raise ValueError(f"Nenhum stream de vídeo em: {self.video_path}")

        fps_parts = video_stream["r_frame_rate"].split("/")
        fps = float(fps_parts[0]) / float(fps_parts[1])
        duration = float(data["format"]["duration"])

        props = {
            "filename": self.video_path.name,
            "width": int(video_stream["width"]),
            "height": int(video_stream["height"]),
            "fps": fps,
            "duration_seconds": duration,
            "duration_minutes": duration / 60,
            "total_frames": int(duration * fps),
            "codec": video_stream["codec_name"],
            "bit_rate_mbps": int(data["format"].get("bit_rate", 0)) / 1_000_000,
            "file_size_gb": float(data["format"]["size"]) / (1024**3),
        }
        logger.info(
            "Vídeo inspecionado: %s — %.1f min, %dx%d, %.2f fps",
            props["filename"],
            props["duration_minutes"],
            props["width"],
            props["height"],
            props["fps"],
        )
        return props

    def save_metadata(self, output_path: str | Path) -> Path:
        """Salva as propriedades como JSON."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(self.properties, f, indent=2)
        logger.info("Metadados do vídeo salvos: %s", out)
        return out


class FrameQualityAnalyzer:
    """
    Calcula métricas de qualidade para uma lista de frames.

    Métricas:
        blur_score  — Variância do Laplaciano (maior = mais nítido)
        brightness  — Brilho médio (0–255)
        contrast    — Desvio padrão de intensidade
    """

    def analyze(self, frame_path: str | Path) -> dict:
        """Analisa um único frame."""
        img = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.warning("Não foi possível ler frame: %s", frame_path)
            return {"blur_score": 0.0, "brightness": 0.0, "contrast": 0.0}
        return {
            "blur_score": float(cv2.Laplacian(img, cv2.CV_64F).var()),
            "brightness": float(img.mean()),
            "contrast": float(img.std()),
        }

    def analyze_batch(self, frame_paths: list[Path]) -> list[dict]:
        """Analisa uma lista de frames, retorna lista de dicts com métricas."""
        results = []
        for p in frame_paths:
            m = self.analyze(p)
            m["frame_path"] = str(p)
            results.append(m)
        logger.info("Qualidade analisada: %d frames", len(results))
        return results

    def summary(self, metrics: list[dict]) -> dict:
        """Estatísticas agregadas sobre um batch de métricas."""
        if not metrics:
            return {}
        blur = np.array([m["blur_score"] for m in metrics])
        bri = np.array([m["brightness"] for m in metrics])
        con = np.array([m["contrast"] for m in metrics])
        return {
            "num_frames": len(metrics),
            "blur": {
                "mean": float(blur.mean()),
                "min": float(blur.min()),
                "max": float(blur.max()),
            },
            "brightness": {
                "mean": float(bri.mean()),
                "min": float(bri.min()),
                "max": float(bri.max()),
            },
            "contrast": {
                "mean": float(con.mean()),
                "min": float(con.min()),
                "max": float(con.max()),
            },
        }
