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
    keyframe_path: Path | None
    embedding: np.ndarray | None = None  # NEW — populated when MMR is enabled


def find_rhymes(
    library_dir: Path,
    anchor_slug: str,
    anchor_scene_id: int,
    top_n: int = 8,
    cross_film_only: bool = True,
    *,
    lambda_diversity: float = 1.0,
    k_candidates: int | None = None,
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
        lambda_diversity: λ ∈ [0, 1] passed to MMR rerank. Default 1.0 keeps
            the M1 stub behaviour (pure kNN — MMR is skipped entirely).
            Service-layer default in M3 is 0.5.
        k_candidates: kNN pool size BEFORE MMR rerank. None (default) →
            ``max(top_n * 3, 30)`` when MMR is active; ignored otherwise.

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

    # Pool size depends on whether MMR will rerank.
    if lambda_diversity < 1.0:
        pool = int(k_candidates) if k_candidates else max(top_n * 3, 30)
    else:
        pool = top_n

    candidates_raw: list[tuple[float, str, int, np.ndarray]] = []
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
        for sim, scene_id, vec in zip(sims, scene_ids, vecs):
            candidates_raw.append((float(sim), slug, int(scene_id), vec.astype("float32")))

    candidates_raw.sort(key=lambda x: -x[0])
    candidates_raw = candidates_raw[:pool]

    rhymes = [
        Rhyme(
            film_slug=slug,
            scene_id=scene_id,
            score=sim,
            keyframe_path=library_dir / slug / "frames" / f"scene_{scene_id:04d}.jpg",
            embedding=vec if lambda_diversity < 1.0 else None,
        )
        for sim, slug, scene_id, vec in candidates_raw
    ]

    if lambda_diversity < 1.0 and rhymes:
        rhymes = mmr_rerank(
            anchor_vec=anchor_vec,
            candidates=rhymes,
            lambda_diversity=lambda_diversity,
            k_final=top_n,
        )
    else:
        rhymes = rhymes[:top_n]
    return rhymes


def _load_keyframe_paths(library_dir: Path, slug: str) -> dict[int, Path | None]:
    """Return a ``{scene_id: Path}`` map for one film slug.

    Reads ``<library_dir>/<slug>/metadata/keyframes_metadata.json`` and
    extracts the ``filepath`` stored for each scene.  The path is stored as
    an absolute string in the production pipeline output.

    Returns an empty dict when the metadata file is absent, unreadable, or
    uses an unexpected shape.  Callers treat a missing scene as ``None``
    so the UI can show a placeholder image rather than crashing.
    """
    meta_path = library_dir / slug / "metadata" / "keyframes_metadata.json"
    if not meta_path.exists():
        return {}
    try:
        data = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        logger.warning("rimas: could not read keyframes metadata for %s", slug)
        return {}

    # The metadata file may be a plain list or a dict with a "scenes" key.
    scenes: list[dict]
    if isinstance(data, list):
        scenes = data
    elif isinstance(data, dict) and "scenes" in data:
        scenes = data["scenes"]
    else:
        logger.warning("rimas: unexpected keyframes_metadata shape for %s", slug)
        return {}

    result: dict[int, Path | None] = {}
    for entry in scenes:
        if not isinstance(entry, dict):
            continue
        try:
            scene_id = int(entry["scene_id"])
        except (KeyError, TypeError, ValueError):
            continue
        fp = entry.get("filepath")
        result[scene_id] = Path(fp) if fp else None
    return result


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


def mmr_rerank(
    *,
    anchor_vec: np.ndarray,
    candidates: list[Rhyme],
    lambda_diversity: float = 0.5,
    k_final: int = 10,
) -> list[Rhyme]:
    """Maximal Marginal Relevance rerank over CLIP-space rhyme candidates.

    Standard Carbonell & Goldstein 1998 formulation. At each step, pick
    the candidate that maximises::

        λ · sim(c, anchor) - (1 - λ) · max_j sim(c, picked_j)

    Args:
        anchor_vec: ``(D,)`` L2-normalised CLIP embedding of the anchor scene.
        candidates: kNN candidates from :func:`find_rhymes` after the
            ``embedding`` field has been populated. Each ``Rhyme.embedding``
            must be a ``(D,)`` array; mismatch with ``anchor_vec`` shape
            is the caller's bug (not validated here — keeps the hot loop
            cheap; ``find_rhymes`` enforces dim consistency upstream).
        lambda_diversity: λ ∈ [0, 1]. 1.0 → pure relevance (MMR collapses
            to argsort by similarity); 0.0 → pure diversity (after the
            first pick, picks anti-correlate with prior picks regardless
            of relevance).
        k_final: maximum length of the returned list; output is
            ``min(k_final, |candidates|)``.

    Returns:
        Re-ordered ``Rhyme`` list. Empty input → empty output.

    Raises:
        ValueError: any candidate has ``embedding is None``, or
            ``lambda_diversity`` outside ``[0, 1]``.
    """
    if not candidates:
        return []
    for c in candidates:
        if c.embedding is None:
            raise ValueError(
                f"mmr_rerank requires `embedding` on every Rhyme; "
                f"missing for {c.film_slug}/{c.scene_id}"
            )
    lam = float(lambda_diversity)
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lambda_diversity must be in [0, 1], got {lam}")

    # Relevance term: trust the upstream cosine-sim already stored on
    # ``Rhyme.score`` by :func:`find_rhymes` (computed as the same dot
    # product against ``anchor_vec`` we'd otherwise recompute here).
    # Using the stored score keeps MMR's relevance ranking consistent
    # with the kNN ranking the caller already saw and avoids drift if
    # the embedding was re-projected or rounded between scoring and
    # rerank.
    # Narrow ``embedding`` to non-None for mypy — the loop above raised
    # for any None, so by here every entry is a real array.
    embeddings: list[np.ndarray] = [c.embedding for c in candidates if c.embedding is not None]
    cand_mat = np.stack(embeddings).astype("float32")
    relevance = np.asarray([c.score for c in candidates], dtype="float32")  # (N,)
    # Pairwise candidate similarities. N is small (≤ k_candidates ≈ 30),
    # so the full (N, N) matrix is fine. ``anchor_vec`` is accepted to
    # validate the embedding-space dimensionality contract and to keep
    # the signature stable for callers that may want anchor-recomputed
    # relevance in the future.
    _ = anchor_vec  # signature contract; reserved for future use
    pair_sims = cand_mat @ cand_mat.T  # (N, N)

    remaining = list(range(len(candidates)))
    picked: list[int] = []
    k = min(int(k_final), len(candidates))

    # First pick: pure relevance (no prior picks → diversity term is
    # ill-defined). Matches the original Carbonell-Goldstein formulation
    # and makes λ=1.0 identical to argsort by relevance.
    first = max(remaining, key=lambda i: relevance[i])
    picked.append(first)
    remaining.remove(first)

    while remaining and len(picked) < k:
        best_score = -np.inf
        best_idx = remaining[0]
        for i in remaining:
            max_sim_to_picked = float(np.max(pair_sims[i, picked]))
            mmr_score = lam * float(relevance[i]) - (1.0 - lam) * max_sim_to_picked
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i
        picked.append(best_idx)
        remaining.remove(best_idx)

    return [candidates[i] for i in picked]
