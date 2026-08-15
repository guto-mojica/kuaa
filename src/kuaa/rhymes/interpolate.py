"""Interpolation between two anchors — retrieval through the space *between*.

The idea
--------
Ordinary retrieval converges: every leg pulls toward the query, and the
result set is the intersection of things that already look like what you
asked for. Interpolation does the opposite. Given two anchors A and B, it
walks the segment between their embeddings and retrieves at points that
neither anchor would surface on its own — a deliberate gap in convergent
method, where the unfilled space between two ideas is the thing being
searched.

Measured on ``jeca_tatu_1959`` with SigLIP2, interpolating
``homem a cavalo`` ↔ ``interior de casa humilde`` (text-text cosine 0.672)
returns 5 of 10 scenes that appear in *neither* endpoint's top-10. The
strongest of them is a family on the porch of a thatched hut with a horse
nearby — precisely the interpolant, and reachable from neither query alone.

The distance condition
----------------------
The operator only opens space when the anchors are far apart. The same
measurement on ``multidao em festa`` ↔ ``paisagem vazia`` (cosine 0.760)
returns 0 of 10 novel scenes at every ``t``: the endpoints are near enough
that the segment between them lies inside both neighbourhoods, and
interpolation degenerates to ordinary search.

That is why :data:`MAX_ANCHOR_COSINE` is a fixed validity gate and not a
user control. It is a failure mode, not a preference — below it there is
nothing between the anchors to find. The dial worth exposing is ``t``, the
position along the segment (see :func:`interpolate_path`).

Deliberately *not* a similarity floor: this module never thresholds result
cosines. ``embeddings.min_similarity`` and ``rimas.threshold`` both ship at
``0.0`` because SigLIP2's score band is too narrow for an absolute floor to
separate signal from noise — real hits and off-distribution noise overlap
in the same 0.2 range. A floor here would reproduce a knob the project has
already found useless.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kuaa.rhymes.algorithm import _load_film_embeddings, _vec_for_scene

logger = logging.getLogger(__name__)

# Above this cosine the two anchors are too close for a meaningful gap.
# Derived from the measurement in the module docstring: 0.672 yields 5/10
# novel scenes, 0.760 yields none.
MAX_ANCHOR_COSINE: float = 0.75

# Default positions along A→B. Excludes 0.0 and 1.0, which are just the
# endpoints — the interesting band is the interior.
DEFAULT_STEPS: tuple[float, ...] = (0.25, 0.5, 0.75)


class AnchorsTooSimilar(ValueError):
    """Raised when two anchors are too close to leave any space between them."""

    def __init__(self, cosine: float) -> None:
        self.cosine = float(cosine)
        super().__init__(
            f"anchors are too similar (cosine {cosine:.3f} >= {MAX_ANCHOR_COSINE}); "
            "interpolation degenerates to ordinary search — pick anchors that "
            "differ more"
        )


@dataclass(frozen=True)
class Interpolant:
    """One scene retrieved from a point between two anchors.

    Attributes:
        film_slug: film the scene belongs to.
        scene_id: the scene.
        score: cosine against the interpolated vector at ``t``.
        t: position along A→B that surfaced it (0 = A, 1 = B).
        novel: True when the scene appears in neither endpoint's own
            top-k — i.e. it exists only in the space between them. This
            is the signal worth surfacing in the UI.
    """

    film_slug: str
    scene_id: int
    score: float
    t: float
    novel: bool


def _normalise(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec)) or 1.0
    return (vec / norm).astype("float32")


def slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Spherical interpolation between two unit vectors.

    Both SigLIP2 image and text embeddings are L2-normalised, so they live
    on the unit sphere. Plain linear interpolation leaves that surface —
    the midpoint of two near-opposite vectors has norm ~0 and renormalising
    it amplifies whatever numerical noise survived. Slerp stays on the
    sphere, so every step is a well-formed query vector and steps are
    evenly spaced in angle rather than bunched near the endpoints.

    Falls back to normalised linear interpolation when the anchors are
    nearly parallel or antiparallel, where the slerp denominator vanishes.
    """
    a = _normalise(a)
    b = _normalise(b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    omega = float(np.arccos(dot))
    sin_omega = float(np.sin(omega))
    if sin_omega < 1e-6:
        return _normalise((1.0 - t) * a + t * b)
    return _normalise(
        (np.sin((1.0 - t) * omega) / sin_omega) * a + (np.sin(t * omega) / sin_omega) * b
    )


def _library_vectors(
    library_dir: Path, slugs: list[str] | None = None
) -> list[tuple[str, np.ndarray, list[int]]]:
    """Load ``(slug, vectors, scene_ids)`` for each indexed film."""
    out: list[tuple[str, np.ndarray, list[int]]] = []
    for film_dir in sorted(Path(library_dir).iterdir()):
        if not film_dir.is_dir():
            continue
        if slugs is not None and film_dir.name not in slugs:
            continue
        loaded = _load_film_embeddings(Path(library_dir), film_dir.name)
        if loaded is None:
            continue
        out.append((film_dir.name, loaded[0], loaded[1]))
    return out


def _top_scenes(
    films: list[tuple[str, np.ndarray, list[int]]], vec: np.ndarray, k: int
) -> list[tuple[tuple[str, int], float]]:
    """Best-scoring ``k`` distinct scenes across the library for one vector.

    Deduped by ``(slug, scene_id)`` keeping the best keyframe, mirroring
    how every other retrieval path in the system collapses the 3-keyframes-
    per-scene index down to scenes.
    """
    best: dict[tuple[str, int], float] = {}
    for slug, vecs, scene_ids in films:
        sims = vecs @ vec
        for row, sid in enumerate(scene_ids):
            if sid < 0:
                continue
            key = (slug, int(sid))
            score = float(sims[row])
            if score > best.get(key, -2.0):
                best[key] = score
    return sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:k]


def anchor_vector_for_scene(library_dir: Path, slug: str, scene_id: int) -> np.ndarray | None:
    """Embedding for a scene anchor, or ``None`` when the film is not indexed."""
    loaded = _load_film_embeddings(Path(library_dir), slug)
    if loaded is None:
        return None
    vec = _vec_for_scene(loaded, scene_id)
    return None if vec is None else _normalise(vec)


def interpolate_path(
    library_dir: Path,
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    *,
    steps: tuple[float, ...] = DEFAULT_STEPS,
    top_k: int = 10,
    slugs: list[str] | None = None,
    enforce_distance: bool = True,
) -> list[Interpolant]:
    """Retrieve at each point along the segment between two anchors.

    Args:
        library_dir: root of the per-film layout.
        vec_a: anchor A embedding (text or image; any L2-normalisable vector).
        vec_b: anchor B embedding.
        steps: positions along A→B to retrieve at. This is the dial worth
            exposing to a curator — sweeping ``t`` walks the space between
            the anchors. Endpoints (0.0 / 1.0) are allowed but return
            ordinary single-anchor results.
        top_k: scenes per step, deduped across the library.
        slugs: restrict to these films; ``None`` searches the whole library.
        enforce_distance: raise :class:`AnchorsTooSimilar` when the anchors
            are closer than :data:`MAX_ANCHOR_COSINE`. Callers running an
            explicit experiment can disable the gate; the UI should not.

    Returns:
        Interpolants ordered by ``t`` then score. ``novel`` marks the ones
        that appear in neither endpoint's own ``top_k`` — the results that
        exist only between the anchors.

    Raises:
        AnchorsTooSimilar: when the gate is on and the anchors are too close.
    """
    a = _normalise(vec_a)
    b = _normalise(vec_b)
    cosine = float(np.dot(a, b))
    if enforce_distance and cosine >= MAX_ANCHOR_COSINE:
        raise AnchorsTooSimilar(cosine)

    films = _library_vectors(library_dir, slugs)
    if not films:
        return []

    # The endpoints' own result sets define what "novel" means: anything
    # either anchor would have found on its own is, by construction, not in
    # the space between them.
    endpoint_keys = {key for key, _ in _top_scenes(films, a, top_k)}
    endpoint_keys |= {key for key, _ in _top_scenes(films, b, top_k)}

    out: list[Interpolant] = []
    for t in steps:
        vec = slerp(a, b, float(t))
        for (slug, sid), score in _top_scenes(films, vec, top_k):
            out.append(
                Interpolant(
                    film_slug=slug,
                    scene_id=sid,
                    score=score,
                    t=float(t),
                    novel=(slug, sid) not in endpoint_keys,
                )
            )
    return out


def novelty_rate(interpolants: list[Interpolant], t: float) -> float:
    """Share of results at ``t`` that neither anchor would have surfaced.

    The diagnostic that tells a curator whether a given pair of anchors is
    productive: high means the gap between them holds material, ~0 means
    the anchors are effectively the same query. Returns 0.0 when nothing
    was retrieved at ``t``.
    """
    at_t = [i for i in interpolants if i.t == t]
    if not at_t:
        return 0.0
    return sum(1 for i in at_t if i.novel) / len(at_t)
