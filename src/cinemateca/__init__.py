"""
Cinemateca AI
~~~~~~~~~~~~~
Sistema de catalogação audiovisual com IA para acervos cinematográficos.

Módulos principais:
    config          — Carregamento e validação de configuração (YAML)
    device          — Detecção de hardware (CPU/CUDA/MPS)
    data_prep       — Inspeção de vídeo e extração de frames (FFmpeg)
    scene_detector  — Detecção de cenas e extração de keyframes (PySceneDetect)
    visual_analyzer — Detecção facial, objetos e ambiente (MTCNN + YOLOv8)
    embeddings      — Embeddings visuais e busca semântica (CLIP)
    models.describer — Geração de metadados descritivos (Moondream 2 GGUF)
    pipeline        — Orquestrador do pipeline completo

Uso rápido:
    from cinemateca.config import load_config, setup_logging
    from cinemateca.pipeline import CatalogPipeline

    cfg = load_config("config/local.yaml")
    setup_logging(cfg)
    pipeline = CatalogPipeline(cfg)
    result = pipeline.run("data/raw/meu_filme.mp4")
    print(result.summary())
"""

__version__ = "0.1.0-alpha"
__author__ = "Cinemateca AI Team"

from cinemateca.config import load_config, setup_logging
from cinemateca.pipeline import CatalogPipeline, PipelineResult

__all__ = [
    "load_config",
    "setup_logging",
    "CatalogPipeline",
    "PipelineResult",
]
