"""Filesystem + URL utilities used across the library + services.

Pure functions: take Path / str / float, return Path / str / float / None.
No FastAPI, no Jinja, no FilmContext dependency.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_json(path: Path) -> list | dict | None:
    """Load a JSON file, or return ``None`` if it does not exist.

    Permissive on return type because callers apply their own
    ``or []`` / ``or {}`` defaulting.
    """
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def keyframe_url(filepath: str | Path, data_dir: Path) -> str | None:
    """Convert a stored keyframe filepath to a ``/media/...`` URL.

    Tries the path as-stored and relative to CWD, returning the first
    that resolves *inside* ``data_dir``; ``None`` otherwise.
    """
    fp = Path(filepath)
    for candidate in (fp, Path.cwd() / fp):
        try:
            rel = candidate.resolve().relative_to(data_dir.resolve())
            return f"/media/{rel.as_posix()}"
        except ValueError:
            continue
    return None


def to_smpte(seconds: float, fps: float = 24.0) -> str:
    """Convert seconds to SMPTE ``HH:MM:SS:FF`` notation."""
    fps_int = max(1, round(fps))
    total_frames = int(seconds * fps)
    ff = total_frames % fps_int
    rest = total_frames // fps_int
    ss = rest % 60
    mm = (rest // 60) % 60
    hh = rest // 3600
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


def derive_fps(kf_meta: list) -> float:
    """Infer original video FPS from keyframe metadata entries.

    Uses the first entry where both ``start_frame`` and ``start_time_s``
    are positive (scene-0 always starts at 0 / 0, useless for derivation).
    Falls back to 24.0 when no suitable entry exists.
    """
    for entry in kf_meta:
        t = float(entry.get("start_time_s") or 0.0)
        f = int(entry.get("start_frame") or 0)
        if t > 0 and f > 0:
            return f / t
    return 24.0
