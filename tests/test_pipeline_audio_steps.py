"""Pipeline integration: audio_extract + audio_embed steps.

Hermetic: ffmpeg + CLAP fully mocked. No model load, no real WAV decode.
"""
from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np


def _seed_film(tmp_path: Path):
    """Create a per-film tmp library with seeded keyframes_metadata.json."""
    slug = "demo"
    film_dir = tmp_path / "library" / slug
    (film_dir / "metadata").mkdir(parents=True)
    (film_dir / "frames" / "scenes" / "keyframes_content").mkdir(parents=True)
    (film_dir / "embeddings").mkdir(parents=True)
    # Two scenes, dup keyframe rows per scene.
    scenes = [
        {"scene_id": 1, "filepath": "k1.jpg", "start_time_s": 0.0,
         "end_time_s": 5.0, "keyframe_id": "k1"},
        {"scene_id": 1, "filepath": "k1b.jpg", "start_time_s": 0.0,
         "end_time_s": 5.0, "keyframe_id": "k1b"},
        {"scene_id": 2, "filepath": "k2.jpg", "start_time_s": 5.0,
         "end_time_s": 12.5, "keyframe_id": "k2"},
    ]
    (film_dir / "metadata" / "keyframes_metadata.json").write_text(
        json.dumps(scenes)
    )
    video = film_dir / "raw" / "demo.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"")
    return slug, film_dir, video


def _build_cfg(tmp_path: Path):
    sn = types.SimpleNamespace
    cfg = sn(
        paths=sn(
            library_dir=tmp_path / "library",
            data_dir=tmp_path / "data",
            raw_dir=tmp_path / "raw",
            frames_dir=tmp_path / "frames",
            metadata_dir=tmp_path / "metadata",
            embeddings_dir=tmp_path / "embeddings",
            models_dir=tmp_path / "models",
            outputs_dir=tmp_path / "out",
            logs_dir=tmp_path / "logs",
        ),
        models=sn(
            image_embedder="clip_openclip",
            face_detector="mtcnn_pytorch",
            object_detector="yolov8",
            scene_describer="moondream_gguf",
            environment_classifier="opencv_heuristic",
            audio_embedder="clap_hf",
        ),
        embeddings=sn(
            model="ViT-B-32", pretrained="openai", batch_size=16,
            filename="keyframe_embeddings.npy",
            mapping_filename="index_mapping.json",
        ),
        audio_embeddings=sn(
            model_id="laion/larger_clap_general",
            batch_size=8,
            chunk_seconds=10.0,
            sample_rate=48000,
        ),
        llm=sn(checkpoint_interval=25, process_limit=None,
               descriptions_filename="scene_descriptions.json",
               tags_filename="scene_tags.json",
               gpu_layers=-1, model_id="x", revision="x"),
        visual_analysis=sn(
            face_detection=sn(enabled=True, min_face_size=20,
                              thresholds=[0.6, 0.7, 0.7]),
            object_detection=sn(enabled=True, model="yolov8n.pt",
                                confidence=0.30),
            environment=sn(enabled=True, brightness_threshold=100,
                           edge_density_threshold=0.05),
        ),
        pipeline=sn(
            steps=sn(
                frame_extraction=False, scene_detection=False,
                visual_analysis=False, embeddings=False,
                llm_description=False,
                audio_extract=True, audio_embed=True,
            ),
            skip_existing=True,
            stop_on_error=False,
        ),
        hardware=sn(device="cpu", force_cpu=True),
    )
    for p in vars(cfg.paths).values():
        Path(p).mkdir(parents=True, exist_ok=True)
    return cfg


def _patch_ffmpeg(monkeypatch):
    """Make subprocess.run create the output WAV without real ffmpeg."""
    def fake_run(cmd, **kwargs):
        out = Path(cmd[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    from cinemateca import audio_extractor
    monkeypatch.setattr(audio_extractor.subprocess, "run", fake_run)


def _patch_clap(monkeypatch, dim=512):
    """Patch ClapHFEmbedder so encode_audio_single returns a unit vector."""
    from cinemateca.models.audio import clap_hf

    def fake_encode_audio_single(self, wav_path):
        # Deterministic unit vector keyed by file stem.
        sid = int(Path(wav_path).stem.split("_")[1])
        v = np.zeros(dim, dtype="float32")
        v[sid % dim] = 1.0
        return v

    monkeypatch.setattr(clap_hf.ClapHFEmbedder, "_load_model", lambda self: None)
    monkeypatch.setattr(
        clap_hf.ClapHFEmbedder, "encode_audio_single", fake_encode_audio_single
    )


def test_audio_extract_step_writes_one_wav_per_scene(monkeypatch, tmp_path):
    from cinemateca.pipeline import CatalogPipeline

    cfg = _build_cfg(tmp_path)
    slug, film_dir, video = _seed_film(tmp_path)
    _patch_ffmpeg(monkeypatch)

    pipe = CatalogPipeline(cfg, slug=slug)
    result = pipe.run_steps(video, ["audio_extract"])

    assert result.ok
    wavs = sorted((film_dir / "audio" / "segments").glob("*.wav"))
    assert [p.name for p in wavs] == ["scene_0001.wav", "scene_0002.wav"]


def test_audio_embed_step_writes_npy_and_mapping(monkeypatch, tmp_path):
    from cinemateca.pipeline import CatalogPipeline

    cfg = _build_cfg(tmp_path)
    slug, film_dir, video = _seed_film(tmp_path)
    _patch_ffmpeg(monkeypatch)
    _patch_clap(monkeypatch)

    pipe = CatalogPipeline(cfg, slug=slug)
    result = pipe.run_steps(video, ["audio_extract", "audio_embed"])
    assert result.ok, [(r.name, r.state, r.error) for r in result.runs]

    npy = film_dir / "audio" / "clap_embeddings.npy"
    mapping = film_dir / "audio" / "audio_mapping.json"
    assert npy.exists()
    assert mapping.exists()

    emb = np.load(npy)
    assert emb.shape == (2, 512)
    assert emb.dtype == np.float32

    m = json.loads(mapping.read_text())
    assert m["total_vectors"] == 2
    assert m["scene_ids"] == [1, 2]
    assert m["model"] == "laion/larger_clap_general"
    assert m["normalized"] is True


def test_audio_embed_blocked_when_audio_extract_missing(monkeypatch, tmp_path):
    """If WAVs aren't on disk and audio_extract isn't in this invocation,
    audio_embed must be blocked (not crash)."""
    from cinemateca.pipeline import CatalogPipeline

    cfg = _build_cfg(tmp_path)
    slug, film_dir, video = _seed_film(tmp_path)
    _patch_clap(monkeypatch)

    pipe = CatalogPipeline(cfg, slug=slug)
    result = pipe.run_steps(video, ["audio_embed"])
    states = {r.name: r.state for r in result.runs}
    assert states["audio_embed"] == "blocked"


def test_audio_extract_blocked_when_metadata_missing(monkeypatch, tmp_path):
    """audio_extract depends on keyframes_metadata.json — block if absent."""
    from cinemateca.pipeline import CatalogPipeline

    cfg = _build_cfg(tmp_path)
    # Create film dir but NOT the metadata file.
    slug = "nometa"
    film_dir = tmp_path / "library" / slug
    (film_dir / "raw").mkdir(parents=True)
    video = film_dir / "raw" / "nometa.mp4"
    video.write_bytes(b"")
    _patch_ffmpeg(monkeypatch)

    pipe = CatalogPipeline(cfg, slug=slug)
    result = pipe.run_steps(video, ["audio_extract"])
    states = {r.name: r.state for r in result.runs}
    assert states["audio_extract"] == "blocked"


def test_audio_extract_skip_existing_is_honoured(monkeypatch, tmp_path):
    """Second invocation with skip_existing=True must report ``skipped``."""
    from cinemateca.pipeline import CatalogPipeline

    cfg = _build_cfg(tmp_path)
    slug, film_dir, video = _seed_film(tmp_path)
    _patch_ffmpeg(monkeypatch)
    _patch_clap(monkeypatch)

    pipe = CatalogPipeline(cfg, slug=slug)
    pipe.run_steps(video, ["audio_extract", "audio_embed"])
    # Second run with all artefacts present.
    result = pipe.run_steps(video, ["audio_extract", "audio_embed"])
    states = {r.name: r.state for r in result.runs}
    assert states["audio_extract"] == "skipped"
    assert states["audio_embed"] == "skipped"


def test_step_order_includes_new_steps_at_end():
    from cinemateca.pipeline import STEP_DEPS, STEP_ORDER

    assert STEP_ORDER == (
        "frame_extraction",
        "scene_detection",
        "visual_analysis",
        "embeddings",
        "llm_description",
        "audio_extract",
        "audio_embed",
    )
    assert STEP_DEPS["audio_extract"] == ("scene_detection",)
    assert STEP_DEPS["audio_embed"] == ("audio_extract",)


def test_audio_embed_errors_with_descriptive_message_when_wav_missing(
    monkeypatch, tmp_path
):
    """If metadata names 3 scenes but only 2 WAVs exist, audio_embed must
    surface a clear error pointing at the first missing path — the gate is
    permissive (one WAV present), so this is the load-bearing check."""
    from cinemateca.pipeline import CatalogPipeline

    cfg = _build_cfg(tmp_path)
    slug, film_dir, video = _seed_film(tmp_path)
    _patch_clap(monkeypatch)

    # Add a third scene to the metadata that has NO matching WAV on disk.
    meta = film_dir / "metadata" / "keyframes_metadata.json"
    scenes = json.loads(meta.read_text())
    scenes.append(
        {"scene_id": 3, "filepath": "k3.jpg", "start_time_s": 12.5,
         "end_time_s": 18.0, "keyframe_id": "k3"}
    )
    meta.write_text(json.dumps(scenes))

    # Hand-seed only WAVs for scenes 1 and 2 so the input gate passes
    # (it only requires ≥1 WAV) but the step's strict check fails.
    segments = film_dir / "audio" / "segments"
    segments.mkdir(parents=True)
    (segments / "scene_0001.wav").write_bytes(b"RIFF")
    (segments / "scene_0002.wav").write_bytes(b"RIFF")

    pipe = CatalogPipeline(cfg, slug=slug)
    result = pipe.run_steps(video, ["audio_embed"])
    run = next(r for r in result.runs if r.name == "audio_embed")
    assert run.state == "error"
    assert "missing" in (run.error or "").lower()
    assert "scene_0003.wav" in (run.error or "")


def test_default_config_has_audio_section_and_defaults():
    from cinemateca.config import load_config

    cfg = load_config()
    assert cfg.models.audio_embedder == "clap_hf"
    assert cfg.audio_embeddings.model_id == "laion/larger_clap_general"
    assert cfg.audio_embeddings.batch_size == 8
    assert cfg.audio_embeddings.chunk_seconds == 10.0
    assert cfg.audio_embeddings.sample_rate == 48000
    # New pipeline flags exist, both off by default (opt-in for M1
    # scaffold; M2 retrieval work flips them on).
    assert cfg.pipeline.steps.audio_extract is False
    assert cfg.pipeline.steps.audio_embed is False
