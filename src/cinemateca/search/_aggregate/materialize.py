"""Hit-dict materialisation for the aggregate pipeline (C1).

Verbatim Phase-4 of the pre-C1 ``aggregate_search``: turn the unified
``ranked`` list of ``((film_slug, scene_id), score)`` pairs into the
``.b-card``-shaped hit dicts the route layer consumes, including the
BM25-only ``iloc[0]`` fallback and the SMPTE timecode build.
"""

from __future__ import annotations

from typing import Any

from cinemateca.library import to_smpte


def materialize_hits(
    ranked: list[tuple[tuple[str, int], float]],
    per_film: dict[str, dict[str, Any]],
    top_k: int,
) -> list[dict]:
    """Materialise the top-``top_k`` ranked keys into hit dicts.

    Keys are already unique ``(film_slug, scene_id)`` so no dedupe pass
    is needed. The keyframe path uses the per-film best-cosine row when
    available (``best_row_by_sid``); a pure BM25-only scene falls back to
    the first kf_df row for that scene_id — deterministic because kf_df
    row order is stable across loads.
    """
    all_hits: list[dict] = []
    for (slug, sid), score in ranked[:top_k]:
        state = per_film.get(slug)
        if state is None:  # defensive — every key came from per_film
            continue
        kf_df = state["kf_df"]
        best_i = state["best_row_by_sid"].get(sid)
        if best_i is not None:
            row = kf_df.iloc[best_i]
        else:
            # BM25-only scene whose cosine was below ``min_similarity``
            # or is otherwise absent from the CLIP-side map. Fall back
            # to the first kf_df row for that scene_id — deterministic
            # because kf_df row order is stable across loads.
            row_mask = kf_df["scene_id"] == sid
            if not row_mask.any():
                continue
            row = kf_df[row_mask].iloc[0]
        meta = state["meta_by_scene"].get(sid)
        start_s = float(meta.get("start_time_s") or 0.0) if meta else 0.0
        timecode = to_smpte(start_s, state["fps"]) if start_s > 0 else ""
        all_hits.append(
            {
                "film_slug": slug,
                "film_title": state["film"].title,
                "scene_id": sid,
                "score": float(score),
                "keyframe_path": str(row["filepath"]),
                "timecode": timecode,
            }
        )
    return all_hits
