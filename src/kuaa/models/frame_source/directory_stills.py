"""
kuaa.models.frame_source.directory_stills
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Frame source for a folder of still images (no video, no scene detection).

Each image becomes a synthetic single-frame "scene": the folder is globbed
(sorted for determinism), every image is converted to 8-bit RGB, downscaled
to ``scene_detection.keyframe_height``, and written as
``scene_NNNN_kf_01.jpg`` into the keyframes directory. The emitted
``keyframes_metadata.json`` uses the exact row schema the video path
produces (with zeroed temporal fields), so visual analysis, embeddings,
description, hybrid search, and cross-image rhymes all work unchanged.

This unlocks cataloguing generated stills (Stable-Diffusion / Flux output
sets) and other non-temporal image corpora without touching downstream code.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from kuaa.models.base import KeyframeManifest

if TYPE_CHECKING:
    from kuaa.config import Settings

logger = logging.getLogger(__name__)

# Raster formats Pillow can open and SigLIP/CLIP can embed. Matched
# case-insensitively against each file's suffix.
_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"})


class DirectoryStillsFrameSource:
    """FrameSource that treats a directory of images as single-frame scenes."""

    def __init__(self, cfg: Settings, device=None) -> None:
        # Device is unused (I/O + resize only); accepted for a uniform
        # factory signature with the model backends.
        self.cfg = cfg
        sd = getattr(cfg, "scene_detection", None)
        self.keyframe_height = int(getattr(sd, "keyframe_height", 480) or 0)

    def _discover(self, source: Path) -> list[Path]:
        """Return sorted image paths directly under ``source``."""
        if not source.exists():
            raise FileNotFoundError(f"Stills source not found: {source}")
        if not source.is_dir():
            raise NotADirectoryError(
                f"directory_stills expects a folder of images, got a file: {source}. "
                "Point --video/source at the directory instead."
            )
        images = sorted(
            p for p in source.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
        )
        if not images:
            raise FileNotFoundError(
                f"No images ({', '.join(sorted(_IMAGE_SUFFIXES))}) found in: {source}"
            )
        return images

    def _write_keyframe(self, src: Path, dest: Path) -> None:
        """Convert ``src`` to 8-bit RGB, downscale to keyframe_height, save JPEG."""
        from PIL import Image

        with Image.open(src) as img:
            rgb = img.convert("RGB")
            h = self.keyframe_height
            if h > 0 and rgb.height != h:
                ratio = h / rgb.height
                new_w = max(1, round(rgb.width * ratio))
                rgb = rgb.resize((new_w, h), Image.Resampling.LANCZOS)
            rgb.save(dest, format="JPEG", quality=95)

    def produce(
        self,
        source: str | Path,
        *,
        keyframes_dir: Path,
        metadata_path: Path,
        cuts_path: Path | None = None,
    ) -> KeyframeManifest:
        """Glob ``source`` for images and emit keyframes + manifest.

        ``cuts_path`` is accepted for signature parity but ignored: still
        sets have no temporal boundaries, so no ``scene_cuts.json`` is
        written (the Pre-processing review UI is video-only).
        """
        source = Path(source)
        keyframes_dir = Path(keyframes_dir)
        keyframes_dir.mkdir(parents=True, exist_ok=True)

        images = self._discover(source)
        rows: list[dict] = []
        keyframes: list[Path] = []
        for idx, src in enumerate(images, start=1):
            keyframe_id = f"scene_{idx:04d}_kf_01"
            dest = keyframes_dir / f"{keyframe_id}.jpg"
            self._write_keyframe(src, dest)
            keyframes.append(dest)
            rows.append(
                {
                    "scene_id": idx,
                    "keyframe_id": keyframe_id,
                    "filepath": str(dest),
                    "start_time_s": 0.0,
                    "end_time_s": 0.0,
                    "duration_s": 0.0,
                    "start_frame": 0,
                    "end_frame": 0,
                    "source_path": str(src),
                }
            )

        metadata_path = Path(metadata_path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)

        stats = {
            "num_scenes": len(rows),
            "total_duration_s": 0.0,
            "mean_s": 0.0,
            "median_s": 0.0,
            "min_s": 0.0,
            "max_s": 0.0,
            "std_s": 0.0,
        }
        logger.info(
            "✓ %d stills catalogados como cenas de frame único em %s",
            len(rows),
            keyframes_dir,
        )
        return {
            "metadata_path": metadata_path,
            "keyframes": keyframes,
            "keyframes_dir": keyframes_dir,
            "stats": stats,
        }
