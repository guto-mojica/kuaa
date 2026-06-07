"""One-shot migration from the v0.3 flat data layout to per-film library.

Usage:
    uv run python scripts/migrate_flat_to_library.py \\
        --flat-root data \\
        --library-dir data/library \\
        --slug jeca_tatu --title "Jeca Tatu" --year 1959

Idempotent in *outcome*: re-running with the same args produces the same
final state. Note: ``shutil.copytree`` unconditionally overwrites every
destination file even if identical, so a re-run on a large library
re-copies every byte. Run once and then leave alone.

Copies (does NOT delete) the flat artefacts. Manual cleanup of the legacy
``data/{raw,frames,metadata,embeddings}/`` dirs is the operator's choice
after they verify the new layout works.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from cinemateca.library import load_registry, register_film

logger = logging.getLogger(__name__)

_SUBDIRS = ["raw", "frames", "metadata", "embeddings"]
_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".m4v", ".webm"}


def migrate_flat_to_library(
    *,
    flat_root: Path,
    library_dir: Path,
    slug: str,
    title: str,
    year: int | None,
) -> None:
    """Copy data/<sub> → data/library/<slug>/<sub> and register the film.

    Behaviour:
      * Idempotent: if the per-film layout already exists and matches the
        registry, do nothing.
      * Copies (``shutil.copytree`` with ``dirs_exist_ok=True``); does NOT
        delete the source. Operator decides whether to clean up.
      * Requires at least one video file in ``flat_root/raw/`` —
        ``FileNotFoundError`` otherwise.
    """
    flat_raw = flat_root / "raw"
    raw_videos = sorted(
        [p for p in flat_raw.iterdir() if p.suffix.lower() in _VIDEO_EXTENSIONS]
        if flat_raw.exists()
        else []
    )
    if not raw_videos:
        raise FileNotFoundError(
            f"No raw video files in {flat_raw}. Accepted extensions: {sorted(_VIDEO_EXTENSIONS)}"
        )
    if len(raw_videos) > 1:
        logger.warning(
            "Multiple videos in %s; picking %s (alphabetical first). Other candidates: %s",
            flat_raw,
            raw_videos[0].name,
            [p.name for p in raw_videos[1:]],
        )

    raw_filename = raw_videos[0].name

    film_dir = library_dir / slug
    film_dir.mkdir(parents=True, exist_ok=True)

    for sub in _SUBDIRS:
        src = flat_root / sub
        dst = film_dir / sub
        if not src.exists():
            logger.warning("Skip absent source: %s", src)
            dst.mkdir(parents=True, exist_ok=True)
            continue
        shutil.copytree(src, dst, dirs_exist_ok=True)
        logger.info("Copied %s → %s", src, dst)

    registry = load_registry(library_dir)
    if slug not in registry:
        register_film(
            library_dir,
            slug=slug,
            title=title,
            year=year,
            raw_filename=raw_filename,
        )
    else:
        logger.info("Slug %s already registered; skipping register_film", slug)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flat-root", type=Path, required=True)
    parser.add_argument("--library-dir", type=Path, required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--year", type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    migrate_flat_to_library(
        flat_root=args.flat_root,
        library_dir=args.library_dir,
        slug=args.slug,
        title=args.title,
        year=args.year,
    )


if __name__ == "__main__":
    main()
