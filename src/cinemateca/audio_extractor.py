"""
cinemateca.audio_extractor
~~~~~~~~~~~~~~~~~~~~~~~~~~
Per-scene audio segment extraction via FFmpeg.

Reads scene boundaries from ``keyframes_metadata.json`` (deduping the
N-rows-per-scene shape down to one row per scene_id) and emits one
mono 16-bit PCM WAV per scene at ``cfg.audio_embeddings.sample_rate``
(default 48 kHz — CLAP's native rate). Filename pattern
``scene_NNNN.wav`` with zero-padded 4-digit scene_id keeps directory
listings sortable.
"""

from __future__ import annotations

import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logger = logging.getLogger(__name__)


def unique_scenes(rows: list[dict]) -> list[dict]:
    """Dedup keyframe-rows by ``scene_id``, keep first occurrence's times.

    keyframes_metadata.json carries N rows per scene (one per keyframe);
    most pipeline steps need one row per scene. Returns rows sorted by
    ``scene_id`` ascending.
    """
    seen: dict[int, dict] = {}
    for row in rows:
        sid = int(row["scene_id"])
        if sid not in seen:
            seen[sid] = {
                "scene_id": sid,
                "start_time_s": float(row["start_time_s"]),
                "end_time_s": float(row["end_time_s"]),
            }
    return [seen[sid] for sid in sorted(seen)]


class SceneAudioExtractor:
    """FFmpeg-driven per-scene audio extractor.

    Args:
        cfg: Effective config (``cfg.audio_embeddings.sample_rate`` +
            ``cfg.pipeline.skip_existing`` are read).
    """

    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self._sample_rate = int(getattr(cfg.audio_embeddings, "sample_rate", 48000))
        self._skip_existing = bool(getattr(cfg.pipeline, "skip_existing", True))

    def extract(
        self,
        video_path: Path,
        scenes: list[dict],
        output_dir: Path,
    ) -> list[Path]:
        """Extract one WAV per unique ``scene_id`` in ``scenes``.

        Args:
            video_path: Source video.
            scenes: Rows from ``keyframes_metadata.json``. Duplicate
                scene_ids are deduped.
            output_dir: Directory to write ``scene_NNNN.wav`` files into.

        Returns:
            WAV paths sorted by scene_id ascending.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        rows = unique_scenes(scenes)
        targets: list[tuple[dict, Path]] = []
        results: list[Path] = []
        for row in rows:
            out_path = output_dir / f"scene_{row['scene_id']:04d}.wav"
            results.append(out_path)
            if self._skip_existing and out_path.exists():
                logger.debug("↷ %s exists, skipping ffmpeg", out_path.name)
                continue
            targets.append((row, out_path))

        if targets:
            max_workers = min(len(targets), os.cpu_count() or 4, 4)
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                # list() forces all to complete and propagates exceptions.
                list(pool.map(lambda t: self._extract_one(video_path, *t), targets))

        logger.info("✓ Áudio por cena: %d WAVs em %s", len(results), output_dir)
        return results

    def _extract_one(self, video_path: Path, row: dict, out_path: Path) -> None:
        # Fast seek: -ss before -i forces ffmpeg to skip-decode to the
        # nearest audio packet rather than decoding from the file start.
        duration = max(0.0, row["end_time_s"] - row["start_time_s"])
        cmd = [
            "ffmpeg",
            "-ss",
            f"{row['start_time_s']}",
            "-t",
            f"{duration}",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(self._sample_rate),
            "-c:a",
            "pcm_s16le",
            "-y",
            "-loglevel",
            "error",
            str(out_path),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            sid = row["scene_id"]
            raise RuntimeError(f"FFmpeg falhou em scene_{sid:04d}:\n{e.stderr}") from e
        except FileNotFoundError as e:
            raise RuntimeError(
                "FFmpeg não encontrado. Instale o FFmpeg: https://ffmpeg.org/download.html"
            ) from e
