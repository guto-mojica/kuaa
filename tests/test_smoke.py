"""
tests/test_smoke.py
~~~~~~~~~~~~~~~~~~~
Smoke tests básicos: verificam que os módulos importam corretamente
e que a configuração funciona, SEM precisar de GPU ou arquivos de vídeo.
"""

import sys
from pathlib import Path

import pytest

# Adicionar src ao path para testes sem instalação
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ─── Config ───────────────────────────────────────────────────────────────────

def test_config_loads_defaults(tmp_path):
    """Config padrão carrega sem erro."""
    from cinemateca.config import load_config
    cfg = load_config(project_root=tmp_path)
    assert cfg.project.name == "Cinemateca AI"
    assert hasattr(cfg, "paths")
    assert hasattr(cfg, "hardware")
    assert hasattr(cfg, "pipeline")


def test_config_resolves_paths(tmp_path):
    """Caminhos relativos são convertidos para absolutos."""
    from cinemateca.config import load_config
    cfg = load_config(project_root=tmp_path)
    # Todos os paths devem ser Path absolutos
    assert cfg.paths.metadata_dir.is_absolute()
    assert cfg.paths.frames_dir.is_absolute()


def test_config_user_override(tmp_path):
    """Config do usuário sobrescreve valores específicos."""
    import yaml

    from cinemateca.config import load_config

    user_cfg = tmp_path / "local.yaml"
    user_cfg.write_text(yaml.dump({
        "scene_detection": {"content_threshold": 42.0}
    }))

    cfg = load_config(str(user_cfg), project_root=tmp_path)
    assert cfg.scene_detection.content_threshold == 42.0
    # Valores não sobrescritos mantêm o default
    assert cfg.scene_detection.min_scene_len == 15


def test_config_setup_logging(tmp_path):
    """setup_logging não levanta exceção."""
    from cinemateca.config import load_config, setup_logging
    cfg = load_config(project_root=tmp_path)
    setup_logging(cfg)  # não deve levantar exceção


# ─── Device ───────────────────────────────────────────────────────────────────

def test_device_cpu_forced():
    """get_device('cpu') sempre retorna CPU."""
    pytest.importorskip("torch")
    from cinemateca.device import get_device
    device = get_device("cpu")
    assert str(device) == "cpu"


def test_device_from_config(tmp_path):
    """device_from_config usa a config corretamente."""
    pytest.importorskip("torch")
    from cinemateca.config import load_config
    from cinemateca.device import device_from_config

    cfg = load_config(project_root=tmp_path)
    device = device_from_config(cfg)
    assert device is not None
    assert hasattr(device, "type")


# ─── Módulos — importação ─────────────────────────────────────────────────────

def test_import_data_prep():
    from cinemateca.data_prep import VideoInspector
    assert VideoInspector is not None


def test_import_scene_detector():
    from cinemateca.scene_detector import SceneDetector
    assert SceneDetector is not None


def test_import_visual_analyzer():
    from cinemateca.visual_analyzer import VisualAnalyzer
    assert VisualAnalyzer is not None


def test_import_embeddings():
    from cinemateca.embeddings import SemanticSearch
    from cinemateca.models.clip.openclip import OpenClipEmbedder
    assert SemanticSearch is not None
    assert OpenClipEmbedder is not None


def test_import_llm_describer():
    from cinemateca.llm_describer import LLMDescriber
    assert LLMDescriber is not None


def test_import_pipeline():
    from cinemateca.pipeline import CatalogPipeline
    assert CatalogPipeline is not None


# ─── LLM parsing — sem modelo ────────────────────────────────────────────────

def test_parse_num_people():
    from cinemateca.llm_describer import _parse_num_people
    assert _parse_num_people("no people visible") == 0
    assert _parse_num_people("two people talking") == 2
    assert _parse_num_people("a man standing") == 1
    assert _parse_num_people("several people in crowd") == -1
    assert _parse_num_people("3 workers") == 3


def test_parse_objects():
    from cinemateca.llm_describer import _parse_objects
    result = _parse_objects("tree, wooden fence, hat, dirt road")
    assert "tree" in result
    assert "wooden fence" in result
    assert len(result) <= 6


def test_generate_tags():
    from cinemateca.llm_describer import _generate_tags
    tags = _generate_tags({
        "location": "exterior",
        "time_of_day": "dia",
        "num_people": 2,
        "objects": ["horse", "dirt road"],
        "setting": "rural field",
    })
    assert "exterior" in tags
    assert "dia" in tags
    assert "duas-pessoas" in tags
    assert "horse" in tags


# ─── FrameQualityAnalyzer — sem vídeo ────────────────────────────────────────

def test_quality_analyzer_missing_file(tmp_path):
    """Arquivo ausente não levanta exceção — retorna zeros."""
    from cinemateca.data_prep import FrameQualityAnalyzer
    analyzer = FrameQualityAnalyzer()
    result = analyzer.analyze(tmp_path / "nonexistent.jpg")
    assert result["blur_score"] == 0.0


# ─── Pipeline — instanciação sem executar ────────────────────────────────────

def test_pipeline_instantiation(tmp_path):
    from cinemateca.config import load_config
    from cinemateca.pipeline import CatalogPipeline
    cfg = load_config(project_root=tmp_path)
    pipeline = CatalogPipeline(cfg)
    assert pipeline is not None


def test_pipeline_result_summary():
    from cinemateca.pipeline import PipelineResult, StepResult
    result = PipelineResult(video_path="test.mp4")
    result.steps.append(StepResult(name="frame_extraction", success=True, duration_s=1.5))
    result.steps.append(StepResult(name="scene_detection", success=False, error="FFmpeg missing"))
    summary = result.summary()
    assert "frame_extraction" in summary
    assert "scene_detection" in summary
    assert "FFmpeg missing" in summary
