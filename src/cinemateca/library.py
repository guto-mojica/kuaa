"""cinemateca.library — Film collection manager."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".m4v", ".webm"}


@dataclass
class Film:
    slug: str
    title: str
    raw_path: Path
    scene_count: int = 0
    is_processed: bool = False


def scan_library(raw_dir: Path, metadata_dir: Path) -> list[Film]:
    """Return all films found in raw_dir, annotated with processing status.

    Currently uses the flat single-film metadata structure. When data is
    reorganized per-film (v0.3.x), update the metadata lookup here.
    """
    if not raw_dir.exists():
        logger.warning("raw_dir not found: %s", raw_dir)
        return []

    kf_path = metadata_dir / "keyframes_metadata.json"
    kf_meta: list[dict] = []
    if kf_path.exists():
        with open(kf_path, encoding="utf-8") as f:
            kf_meta = json.load(f)

    films: list[Film] = []
    for video_path in sorted(raw_dir.iterdir()):
        if video_path.suffix.lower() not in _VIDEO_EXTENSIONS:
            continue
        slug = video_path.stem.lower().replace(" ", "_")
        is_processed = bool(kf_meta)
        films.append(
            Film(
                slug=slug,
                title=video_path.stem.replace("_", " ").title(),
                raw_path=video_path,
                scene_count=len(kf_meta) if is_processed else 0,
                is_processed=is_processed,
            )
        )
        logger.debug("Film: %s (processed=%s, scenes=%d)", slug, is_processed, len(kf_meta))

    return films
