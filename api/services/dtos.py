"""Domain transfer objects for API service layer.

DTOs carry only the data the API layer needs to render a response — no
model logic, no file I/O.  Service functions build these objects and
return them; route handlers consume them.  This cleanly separates the
shape of an API response from the internal structure of domain models.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SceneDTO:
    """Lightweight scene representation for API responses."""

    scene_id: int
    keyframe_path: str
    start_time_s: float
    end_time_s: float
    description: str = ""
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class FilmDTO:
    """Lightweight film representation for API responses."""

    slug: str
    title: str
    scene_count: int
    is_processed: bool
    year: int | None = None
