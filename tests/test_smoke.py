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
    from kuaa.config import load_config

    cfg = load_config(project_root=tmp_path)
    assert cfg.project.name == "KUAA"
    assert hasattr(cfg, "paths")
    assert hasattr(cfg, "hardware")
    assert hasattr(cfg, "pipeline")


def test_config_resolves_paths(tmp_path):
    """Caminhos relativos são convertidos para absolutos."""
    from kuaa.config import load_config

    cfg = load_config(project_root=tmp_path)
    # Todos os paths devem ser Path absolutos
    assert cfg.paths.metadata_dir.is_absolute()
    assert cfg.paths.frames_dir.is_absolute()


def test_config_user_override(tmp_path):
    """Config do usuário sobrescreve valores específicos."""
    import yaml

    from kuaa.config import load_config

    user_cfg = tmp_path / "local.yaml"
    user_cfg.write_text(yaml.dump({"scene_detection": {"content_threshold": 42.0}}))

    cfg = load_config(str(user_cfg), project_root=tmp_path)
    assert cfg.scene_detection.content_threshold == 42.0
    # Valores não sobrescritos mantêm o default
    assert cfg.scene_detection.min_scene_len == 15


def test_config_setup_logging(tmp_path):
    """setup_logging não levanta exceção."""
    from kuaa.config import load_config, setup_logging

    cfg = load_config(project_root=tmp_path)
    setup_logging(cfg)  # não deve levantar exceção


# ─── Device ───────────────────────────────────────────────────────────────────


def test_device_cpu_forced():
    """get_device('cpu') sempre retorna CPU."""
    pytest.importorskip("torch")
    from kuaa.device import get_device

    device = get_device("cpu")
    assert str(device) == "cpu"


def test_device_from_config(tmp_path):
    """device_from_config usa a config corretamente."""
    pytest.importorskip("torch")
    from kuaa.config import load_config
    from kuaa.device import device_from_config

    cfg = load_config(project_root=tmp_path)
    device = device_from_config(cfg)
    assert device is not None
    assert hasattr(device, "type")


# ─── Módulos — importação ─────────────────────────────────────────────────────


def test_import_data_prep():
    from kuaa.data_prep import VideoInspector

    assert VideoInspector is not None


def test_import_scene_detector():
    from kuaa.scene_detector import SceneDetector

    assert SceneDetector is not None


def test_import_visual_analyzer():
    from kuaa.visual_analyzer import VisualAnalyzer

    assert VisualAnalyzer is not None


def test_import_embeddings():
    from kuaa.embeddings import SemanticSearch
    from kuaa.models.clip.openclip import OpenClipEmbedder

    assert SemanticSearch is not None
    assert OpenClipEmbedder is not None


def test_import_describer():
    from kuaa.models.describer.gguf import MoondreamGGUFDescriber

    assert MoondreamGGUFDescriber is not None


def test_import_describer_transformers():
    from kuaa.models.describer.transformers_hf import (
        MoondreamTransformersDescriber,
    )

    assert MoondreamTransformersDescriber is not None


def test_import_pipeline():
    from kuaa.pipeline import CatalogPipeline

    assert CatalogPipeline is not None


# ─── LLM parsing — sem modelo ────────────────────────────────────────────────

# The parser tests below encode how ONE describer phrases its answers:
# "2 people, one holding a gun" (a digit, then "one" as a pronoun), "person"
# and "dark background" listed as objects. That phrasing was observed on the
# library's stored _raw_responses under this revision and these prompts. A
# different model, revision, or prompt may phrase the same facts differently
# and walk past every rule the parser has — silently, since nothing raises.
_OBSERVED_DESCRIBER_REVISION = "2025-01-09"
_OBSERVED_PROMPTS = {
    "people_and_action": (
        "How many people are visible and what are they doing? "
        "Answer briefly, e.g.: 2 people talking."
    ),
    "objects": "List the most notable objects in this scene, comma-separated. Maximum 6 items.",
}


def test_parser_rules_are_pinned_to_the_describer_that_produced_them():
    """Fail loudly when the describer moves out from under the parser tests.

    On failure: re-run the parser over stored raw answers from films described
    with the new model/prompt, check the phrasings the tests assume still
    hold, then update the pins here in the same commit.
    """
    from kuaa.models.describer._common import PROMPTS
    from kuaa.models.manifest import get_card

    for backend in ("moondream_transformers", "moondream_gguf"):
        assert get_card(backend).revision == _OBSERVED_DESCRIBER_REVISION, (
            f"{backend} revision changed — the parser rules in this file were "
            f"calibrated on {_OBSERVED_DESCRIBER_REVISION}; re-verify them"
        )
    for field, text in _OBSERVED_PROMPTS.items():
        assert PROMPTS[field][0] == text, (
            f"the {field!r} prompt changed — the parser reads the answer shape "
            "this prompt elicits; re-verify the rules and update the pin"
        )


def test_parse_objects_drops_people_anatomy_and_picture_plane():
    from kuaa.models.describer._common import _parse_objects

    raw = "Rope, person, human head, dark background, boat, red, text, hammock"
    assert _parse_objects(raw) == ["rope", "boat", "text", "hammock"]
    # A modifier does not rescue a stop head noun; a role noun is kept.
    assert _parse_objects("textured surface, astronaut, climber") == ["astronaut", "climber"]


def test_parse_num_people():
    from kuaa.models.describer._common import _parse_num_people

    assert _parse_num_people("no people visible") == 0
    assert _parse_num_people("two people talking") == 2
    assert _parse_num_people("a man standing") == 1
    assert _parse_num_people("several people in crowd") == -1
    assert _parse_num_people("3 workers") == 3
    # A digit is the count; a later number word is a pronoun. Observed on
    # jangada_1949: 6 of 47 keyframes tagged pessoa-unica off answers like
    # these, because "one" was matched before the digit was looked for.
    assert _parse_num_people("2 people, one holding a gun") == 2
    assert _parse_num_people("2 people are visible, one is holding a clapperboard") == 2
    assert _parse_num_people("someone walking") == -1


def test_parse_objects():
    from kuaa.models.describer._common import _parse_objects

    result = _parse_objects("tree, wooden fence, hat, dirt road")
    assert "tree" in result
    assert "wooden fence" in result
    assert len(result) <= 6


def test_generate_tags():
    from kuaa.models.describer._common import _generate_tags

    tags = _generate_tags(
        {
            "location": "exterior",
            "time_of_day": "dia",
            "num_people": 2,
            "objects": ["horse", "dirt road"],
            "setting": "rural field",
        }
    )
    assert "exterior" in tags
    assert "dia" in tags
    assert "duas-pessoas" in tags
    assert "horse" in tags


# ─── FrameQualityAnalyzer — sem vídeo ────────────────────────────────────────


def test_quality_analyzer_missing_file(tmp_path):
    """Arquivo ausente não levanta exceção — retorna zeros."""
    from kuaa.data_prep import FrameQualityAnalyzer

    analyzer = FrameQualityAnalyzer()
    result = analyzer.analyze(tmp_path / "nonexistent.jpg")
    assert result["blur_score"] == 0.0


# ─── Pipeline — instanciação sem executar ────────────────────────────────────


def test_pipeline_instantiation(tmp_path):
    from kuaa.config import load_config
    from kuaa.pipeline import CatalogPipeline

    cfg = load_config(project_root=tmp_path)
    pipeline = CatalogPipeline(cfg)
    assert pipeline is not None


def test_pipeline_result_summary():
    from kuaa.pipeline import PipelineResult, StepResult

    result = PipelineResult(video_path="test.mp4")
    result.steps.append(StepResult(name="frame_extraction", success=True, duration_s=1.5))
    result.steps.append(StepResult(name="scene_detection", success=False, error="FFmpeg missing"))
    summary = result.summary()
    assert "frame_extraction" in summary
    assert "scene_detection" in summary
    assert "FFmpeg missing" in summary


def test_steps_alias_maps_to_full_names():
    import click

    from kuaa.__main__ import _resolve_steps

    assert _resolve_steps("llm") == {"llm_description"}
    assert _resolve_steps("scenes,visual") == {
        "scene_detection",
        "visual_analysis",
    }
    assert _resolve_steps("llm_description") == {"llm_description"}  # full name OK
    # In the Typer CLI, bad --steps values raise ``typer.BadParameter`` which
    # subclasses ``click.UsageError``. The contract here is "unknown step
    # surfaces an inline CLI error, not a stack trace" — checking the
    # subclass keeps the regression coverage without binding to typer.
    with pytest.raises(click.UsageError):
        _resolve_steps("bogus")
