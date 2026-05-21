"""Cross-film visual rhymes via cosine kNN on CLIP keyframe embeddings.

Minimum-viable backend for the Rimas Visuais (visual rhymes) feature.
The full M3 stack — CLIP × CLAP fusion + MMR diversity + cross-encoder
rerank — replaces this implementation, but the public surface
(``find_rhymes`` returning ``list[Rhyme]``) is intended to stay stable.

Per-film embeddings are expected at::

    <library_dir>/<slug>/embeddings/keyframe_embeddings.npy
    <library_dir>/<slug>/embeddings/index_mapping.json

The mapping file's scene-id list can be encoded in either of two shapes
the existing pipeline produces:

  * synthetic / test fixtures write ``"scene_ids": [1, 2, 3, ...]``
    — one int per embedding row;
  * the real production pipeline (PySceneDetect → CLIP) writes
    ``"keyframe_paths": ["<...>-Scene-001-01.jpg", ...]`` — one filename
    per embedding row, with the scene number embedded in the
    ``Scene-NNN-MM`` portion of the basename.

:func:`_extract_scene_ids` accepts either shape transparently, so the
caller does not need to know which producer wrote the file. Any missing
file or unparseable mapping degrades gracefully to ``[]`` so the caller
can render an empty-state UI without raising.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ``<title>-Scene-NNN-MM.jpg`` — PySceneDetect's keyframe filename pattern.
# We only need the scene number (the ``MM`` suffix is the keyframe index
# within the scene; multiple keyframes per scene collapse to one scene
# id, so duplicate scene ids in the returned list are expected and
# correct — they line up row-for-row with the embeddings matrix).
_SCENE_NUM_RE = re.compile(r"[Ss]cene[-_](\d+)")


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


def _extract_scene_ids(mapping: dict) -> list[int]:
    """Return one scene id per embedding row from either index_mapping shape.

    The mapping dict comes from ``index_mapping.json``. Two shapes exist
    in the wild:

      1. **Synthetic / test shape** — ``{"scene_ids": [1, 2, 3, ...]}``.
         Returned verbatim (coerced to ``int``).
      2. **Production shape** — ``{"keyframe_paths": ["<...>-Scene-001-01.jpg",
         ...]}``. The scene number is parsed out of each filename via
         :data:`_SCENE_NUM_RE`. Rows whose filename does not match the
         pattern are emitted as ``-1`` so the index keeps its row-count
         alignment with the embeddings matrix; the lookup in
         :func:`_vec_for_scene` will simply never resolve to a row
         tagged ``-1``.

    Returns ``[]`` when neither key is present — :func:`_load_film_embeddings`
    treats that as a corrupt mapping and returns ``None`` to the caller.
    """
    if "scene_ids" in mapping:
        return [int(sid) for sid in mapping["scene_ids"]]
    if "keyframe_paths" in mapping:
        scene_ids: list[int] = []
        for path in mapping["keyframe_paths"]:
            match = _SCENE_NUM_RE.search(str(path))
            scene_ids.append(int(match.group(1)) if match else -1)
        return scene_ids
    return []


def _load_film_embeddings(library_dir: Path, slug: str) -> tuple[np.ndarray, list[int]] | None:
    """Load ``(vectors, scene_ids)`` for one film, or ``None`` if absent.

    Mapping-shape flexibility lives in :func:`_extract_scene_ids` — this
    function only enforces row-count alignment between the embeddings
    matrix and the derived scene-id list. A mismatch (or a mapping that
    declares neither known shape) returns ``None`` so the caller falls
    back to the empty-state UI.
    """
    emb_path = library_dir / slug / "embeddings" / "keyframe_embeddings.npy"
    map_path = library_dir / slug / "embeddings" / "index_mapping.json"
    if not (emb_path.exists() and map_path.exists()):
        return None
    vecs: np.ndarray = np.load(emb_path)
    mapping = json.loads(map_path.read_text())
    scene_ids = _extract_scene_ids(mapping)
    if not scene_ids:
        logger.warning(
            "rimas: %s index_mapping.json has neither 'scene_ids' nor "
            "'keyframe_paths' — skipping film",
            slug,
        )
        return None
    if len(scene_ids) != int(vecs.shape[0]):
        logger.warning(
            "rimas: %s embeddings/mapping row mismatch (%d vs %d) — skipping film",
            slug,
            int(vecs.shape[0]),
            len(scene_ids),
        )
        return None
    return vecs, scene_ids


def _vec_for_scene(film: tuple[np.ndarray, list[int]], scene_id: int) -> np.ndarray | None:
    """Look up the embedding row for ``scene_id``; ``None`` if not present."""
    vecs, scene_ids = film
    try:
        idx = scene_ids.index(scene_id)
    except ValueError:
        return None
    return vecs[idx]
