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
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


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
            scenes: Rows from ``keyframes_metadata.json``. Each row must
                carry ``scene_id``, ``start_time_s``, ``end_time_s``.
                Duplicate scene_ids are deduped (keyframes_metadata has
                N rows per scene).
            output_dir: Directory to write ``scene_NNNN.wav`` files into.

        Returns:
            Sorted list of WAV paths (one per unique scene_id, ascending
            by scene_id).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Dedup by scene_id, preserving the first occurrence's times.
        unique: dict[int, dict] = {}
        for row in scenes:
            sid = int(row["scene_id"])
            if sid not in unique:
                unique[sid] = {
                    "scene_id": sid,
                    "start_time_s": float(row["start_time_s"]),
                    "end_time_s": float(row["end_time_s"]),
                }

        results: list[Path] = []
        for sid in sorted(unique):
            row = unique[sid]
            out_path = output_dir / f"scene_{sid:04d}.wav"

            if self._skip_existing and out_path.exists():
                logger.debug("↷ scene_%04d.wav exists, skipping ffmpeg", sid)
                results.append(out_path)
                continue

            cmd = [
                "ffmpeg",
                "-ss",
                f"{row['start_time_s']}",
                "-to",
                f"{row['end_time_s']}",
                "-i",
                str(video_path),
                "-vn",  # no video
                "-ac",
                "1",  # mono
                "-ar",
                str(self._sample_rate),
                "-c:a",
                "pcm_s16le",  # 16-bit PCM
                "-y",  # overwrite if force
                "-loglevel",
                "error",
                str(out_path),
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    f"FFmpeg falhou em scene_{sid:04d}:\n{e.stderr}"
                ) from e
            except FileNotFoundError as e:
                raise RuntimeError(
                    "FFmpeg não encontrado. Instale o FFmpeg: "
                    "https://ffmpeg.org/download.html"
                ) from e
            results.append(out_path)

        logger.info("✓ Áudio por cena: %d WAVs em %s", len(results), output_dir)
        return results
