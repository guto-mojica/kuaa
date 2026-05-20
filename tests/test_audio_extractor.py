"""SceneAudioExtractor unit tests — FFmpeg invocation fully mocked, hermetic."""

from __future__ import annotations

import types
from pathlib import Path

import pytest


def _cfg(sample_rate=48000, skip_existing=True):
    sn = types.SimpleNamespace
    return sn(
        audio_embeddings=sn(
            model_id="laion/larger_clap_general",
            batch_size=8,
            chunk_seconds=10.0,
            sample_rate=sample_rate,
        ),
        pipeline=sn(skip_existing=skip_existing, stop_on_error=False),
    )


def _scenes():
    return [
        {"scene_id": 1, "start_time_s": 0.0, "end_time_s": 5.0, "filepath": "k1.jpg"},
        {
            "scene_id": 1,
            "start_time_s": 0.0,
            "end_time_s": 5.0,
            "filepath": "k2.jpg",
        },  # dup row, same scene_id
        {"scene_id": 2, "start_time_s": 5.0, "end_time_s": 12.5, "filepath": "k3.jpg"},
    ]


def test_extract_dedups_by_scene_id_and_writes_one_wav_per_scene(monkeypatch, tmp_path):
    from cinemateca import audio_extractor

    called = []

    def fake_run(cmd, **kwargs):
        # Simulate ffmpeg producing the output WAV.
        out = Path(cmd[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")
        called.append(cmd)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(audio_extractor.subprocess, "run", fake_run)

    video = tmp_path / "fake.mp4"
    video.write_bytes(b"")  # extractor doesn't read it; ffmpeg is mocked

    out_dir = tmp_path / "out"
    extractor = audio_extractor.SceneAudioExtractor(_cfg())
    wavs = extractor.extract(video, _scenes(), out_dir)

    assert [p.name for p in wavs] == ["scene_0001.wav", "scene_0002.wav"]
    assert all(p.exists() for p in wavs)
    assert len(called) == 2  # one ffmpeg call per unique scene_id


def test_extract_skips_existing_when_skip_existing_true(monkeypatch, tmp_path):
    from cinemateca import audio_extractor

    called = []

    def fake_run(cmd, **kwargs):
        out = Path(cmd[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"RIFF")
        called.append(cmd)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(audio_extractor.subprocess, "run", fake_run)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "scene_0001.wav").write_bytes(b"RIFF")  # pre-existing

    extractor = audio_extractor.SceneAudioExtractor(_cfg())
    wavs = extractor.extract(tmp_path / "v.mp4", _scenes(), out_dir)

    assert len(wavs) == 2
    # Only scene 2 should have triggered ffmpeg (scene 1 was on disk).
    assert len(called) == 1
    # The scene-1 path is still returned (it exists from before).
    assert (out_dir / "scene_0001.wav").exists()


def test_extract_emits_correct_ffmpeg_command(monkeypatch, tmp_path):
    from cinemateca import audio_extractor

    captured = []

    def fake_run(cmd, **kwargs):
        captured.append(cmd)
        Path(cmd[-1]).parent.mkdir(parents=True, exist_ok=True)
        Path(cmd[-1]).write_bytes(b"RIFF")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(audio_extractor.subprocess, "run", fake_run)

    video = tmp_path / "in.mp4"
    extractor = audio_extractor.SceneAudioExtractor(_cfg(sample_rate=48000))
    extractor.extract(video, _scenes()[:1], tmp_path / "out")

    cmd = captured[0]
    assert cmd[0] == "ffmpeg"
    assert "-ss" in cmd and "0.0" in cmd
    assert "-to" in cmd and "5.0" in cmd
    assert "-ac" in cmd and "1" in cmd  # mono
    assert "-ar" in cmd and "48000" in cmd  # 48 kHz
    assert "-c:a" in cmd and "pcm_s16le" in cmd  # PCM16
    assert str(video) in cmd
    assert cmd[-1].endswith("scene_0001.wav")


def test_extract_propagates_ffmpeg_failure(monkeypatch, tmp_path):
    from cinemateca import audio_extractor

    def boom(cmd, **kwargs):
        import subprocess as sp

        raise sp.CalledProcessError(returncode=1, cmd=cmd, stderr="codec error")

    monkeypatch.setattr(audio_extractor.subprocess, "run", boom)
    extractor = audio_extractor.SceneAudioExtractor(_cfg(skip_existing=False))
    with pytest.raises(RuntimeError, match="FFmpeg falhou"):
        extractor.extract(tmp_path / "v.mp4", _scenes()[:1], tmp_path / "out")


def test_extract_raises_friendly_error_when_ffmpeg_binary_missing(monkeypatch, tmp_path):
    from cinemateca import audio_extractor

    def boom(cmd, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", "ffmpeg")

    monkeypatch.setattr(audio_extractor.subprocess, "run", boom)
    extractor = audio_extractor.SceneAudioExtractor(_cfg(skip_existing=False))
    with pytest.raises(RuntimeError, match="FFmpeg não encontrado"):
        extractor.extract(tmp_path / "v.mp4", _scenes()[:1], tmp_path / "out")
