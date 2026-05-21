"""Cross-film visual rhymes via cosine kNN on CLIP keyframe embeddings.

Minimum-viable backend for the Rimas Visuais (visual rhymes) feature.
The full M3 stack — CLIP × CLAP fusion + MMR diversity + cross-encoder
rerank — replaces this implementation, but the public surface
(``find_rhymes`` returning ``list[Rhyme]``) is intended to stay stable.

Per-film embeddings are expected at::

    <library_dir>/<slug>/embeddings/keyframe_embeddings.npy
    <library_dir>/<slug>/embeddings/index_mapping.json

The mapping file must contain a ``scene_ids`` list whose length matches the
number of embedding rows. Any missing file degrades gracefully to ``[]``
so the caller can render an empty-state UI without raising.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Rhyme:
    """A single cross-film visual neighbour of an anchor keyframe."""

    film_slug: str
    scene_id: int
    score: float
    keyframe_path: Path


def find_rhymes(
    library_dir: Path,
    anchor_slug: str,
    anchor_scene_id: int,
    top_n: int = 8,
    cross_film_only: bool = True,
) -> list[Rhyme]:
    """Top-N cosine neighbours of an anchor keyframe across the library.

    Args:
        library_dir: Root of ``data/library``-style per-film directories.
        anchor_slug: Film slug the anchor keyframe belongs to.
        anchor_scene_id: Scene id of the anchor keyframe inside ``anchor_slug``.
        top_n: Maximum number of neighbours to return.
        cross_film_only: When ``True`` (default), candidates from
            ``anchor_slug`` itself are excluded — this is the product
            constraint for the Rimas Visuais tab.

    Returns:
        Ranked ``Rhyme`` list, longest = ``top_n``. Returns ``[]`` if the
        anchor index is missing, the anchor scene is not in the index, or
        no other film has embeddings yet — callers render empty state.
    """
    anchor = _load_film_embeddings(library_dir, anchor_slug)
    if anchor is None:
        logger.info("rimas: anchor %s has no embeddings", anchor_slug)
        return []
    anchor_vec = _vec_for_scene(anchor, anchor_scene_id)
    if anchor_vec is None:
        logger.info(
            "rimas: scene %s not found in %s embeddings index",
            anchor_scene_id,
            anchor_slug,
        )
        return []

    if not library_dir.exists():
        return []

    candidates: list[tuple[float, str, int]] = []
    for film_dir in sorted(library_dir.iterdir()):
        if not film_dir.is_dir():
            continue
        slug = film_dir.name
        if cross_film_only and slug == anchor_slug:
            continue
        film = _load_film_embeddings(library_dir, slug)
        if film is None:
            continue
        vecs, scene_ids = film
        sims = vecs @ anchor_vec
        for sim, scene_id in zip(sims, scene_ids):
            candidates.append((float(sim), slug, int(scene_id)))

    candidates.sort(key=lambda x: -x[0])
    return [
        Rhyme(
            film_slug=slug,
            scene_id=scene_id,
            score=sim,
            keyframe_path=library_dir / slug / "frames" / f"scene_{scene_id:04d}.jpg",
        )
        for sim, slug, scene_id in candidates[:top_n]
    ]


def _load_film_embeddings(library_dir: Path, slug: str) -> tuple[np.ndarray, list[int]] | None:
    """Load ``(vectors, scene_ids)`` for one film, or ``None`` if absent."""
    emb_path = library_dir / slug / "embeddings" / "keyframe_embeddings.npy"
    map_path = library_dir / slug / "embeddings" / "index_mapping.json"
    if not (emb_path.exists() and map_path.exists()):
        return None
    vecs: np.ndarray = np.load(emb_path)
    mapping = json.loads(map_path.read_text())
    scene_ids = [int(sid) for sid in mapping["scene_ids"]]
    return vecs, scene_ids


def _vec_for_scene(film: tuple[np.ndarray, list[int]], scene_id: int) -> np.ndarray | None:
    """Look up the embedding row for ``scene_id``; ``None`` if not present."""
    vecs, scene_ids = film
    try:
        idx = scene_ids.index(scene_id)
    except ValueError:
        return None
    return vecs[idx]
