"""CLIP-only search verbs.

This module hosts the two pure CLIP search dispatchers extracted from
``api/services/search.py`` during P1 (T9). They are intentionally sync —
the caller (an HTTP route or the hybrid dispatcher) is responsible for
running them in a thread executor if it needs to keep the event loop
free. Keeping the verbs sync makes the module framework-agnostic and
unit-testable without an event loop.

Contract — preserved verbatim from the previous home in
``api.services.search``:

  * ``search_text`` takes a loaded :class:`SearchIndex`, asks the
    underlying :class:`SemanticSearch` for ``top_k * 4`` raw hits (4× is
    the configured ``keyframes_per_scene`` ceiling), optionally floors
    by ``min_similarity``, then dedupes by ``scene_id`` and trims to
    ``top_k`` *scenes*. The wider raw window is required so the
    post-dedupe slice still has ``top_k`` distinct scenes to choose
    from when the index has multiple keyframes per scene (Phase-1
    density fix).
  * ``search_image`` follows the same widen-then-dedupe pattern. The
    RAW tag-index is forwarded to :meth:`SemanticSearch.combined`
    unchanged — that method self-normalises (Phase 1c).

The ``SemanticSearch`` import is hoisted to module scope (the legacy
home did a per-call ``from cinemateca.embeddings import …``). One
existing test (``test_search_service.py::TestSceneDedup
::test_search_image_dedupes_by_scene_id``) monkeypatches
``cinemateca.embeddings.SemanticSearch`` to stub out the JPEG-loading
path; after this hoist the test patches
``cinemateca.search.clip.SemanticSearch`` instead — same effect,
adjusted target. The hoist is the deliberate plan-directed change.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from cinemateca.embeddings import SemanticSearch
from cinemateca.search.cache import SearchIndex

logger = logging.getLogger(__name__)


def search_text(
    index: SearchIndex,
    query: str,
    tags: list[str],
    tag_index: dict,
    top_k: int,
    min_similarity: float = 0.0,
) -> pd.DataFrame:
    """Run a text (optionally tag-filtered) semantic search.

    Mirrors the prior route logic exactly: with ``tags`` it calls
    ``SemanticSearch.combined`` passing the RAW merged ``tag_index``
    (``combined`` self-normalizes — Phase 1c contract preserved); without
    tags it calls ``by_text``. Caller (route) is responsible for running
    this in an executor — kept sync here so the service stays
    framework-agnostic and unit-testable without an event loop.

    ``min_similarity`` post-filters the result DataFrame (CLIP returns
    top-K unconditionally, so unrelated queries surface noise scenes;
    the threshold drops anything below the cosine floor). 0.0 disables
    the filter (default for back-compat with unit tests).
    """
    # Callers are expected to check index.ok before calling; these asserts
    # narrow the Optional fields for mypy (index.ok guarantees non-None).
    assert index.embeddings is not None
    assert index.kf_df is not None
    assert index.embedder is not None
    searcher = SemanticSearch(index.embeddings, index.kf_df, index.embedder)
    # The underlying searcher returns the global top-K by similarity; with
    # multiple keyframes per scene that top-K may concentrate inside one
    # scene's keyframe block, starving other scenes. Ask for a wider
    # window (top_k * kf_per_scene) so the post-dedupe top-K still has
    # ``top_k`` distinct scenes to choose from. The wider window only
    # affects ranking, not embedding cost.
    raw_k = top_k * 4  # 4× is the configured ceiling for keyframes_per_scene
    if tags:
        df = searcher.combined(query, tags, tag_index, raw_k)
    else:
        df = searcher.by_text(query, raw_k)
    n_raw = len(df)
    top_raw = float(df["similarity"].iloc[0]) if n_raw and "similarity" in df.columns else 0.0
    if min_similarity > 0.0 and not df.empty and "similarity" in df.columns:
        df = df[df["similarity"] >= min_similarity].reset_index(drop=True)
    # Dedupe by scene_id (Phase-1 density fix). The DataFrame is already
    # ordered by similarity descending, so ``drop_duplicates`` keeps the
    # first occurrence per scene = the best-matching keyframe of that
    # scene. Trim to ``top_k`` AFTER dedup so the UI gets the requested
    # number of *scenes*, not keyframes.
    n_after_floor = len(df)
    if not df.empty and "scene_id" in df.columns:
        df = df.drop_duplicates(subset="scene_id", keep="first").reset_index(drop=True)
    df = df.head(top_k).reset_index(drop=True)
    logger.info(
        "search_text: query=%r top_k=%d tags=%s min_sim=%.3f "
        "raw_hits=%d top_score=%.3f kept_after_floor=%d dedup_kept=%d",
        query,
        top_k,
        tags or None,
        min_similarity,
        n_raw,
        top_raw,
        n_after_floor,
        len(df),
    )
    return df


def search_image(index: SearchIndex, image_path: Path | str, top_k: int) -> pd.DataFrame:
    """Run an image-similarity semantic search (sync; see :func:`search_text`).

    Applies the same scene_id dedupe as :func:`search_text` so the UI
    receives at most one card per scene, displaying the best-matching
    keyframe (rather than three near-duplicate rows from the same shot).
    """
    # Callers are expected to check index.ok before calling; these asserts
    # narrow the Optional fields for mypy (index.ok guarantees non-None).
    assert index.embeddings is not None
    assert index.kf_df is not None
    assert index.embedder is not None
    searcher = SemanticSearch(index.embeddings, index.kf_df, index.embedder)
    df = searcher.by_image(image_path, top_k * 4)
    if not df.empty and "scene_id" in df.columns:
        df = df.drop_duplicates(subset="scene_id", keep="first").reset_index(drop=True)
    return df.head(top_k).reset_index(drop=True)
