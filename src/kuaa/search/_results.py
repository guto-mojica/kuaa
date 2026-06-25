"""DataFrame → template-dict conversion. Private to the search package.

Extracted from ``api/services/search.py::results_to_dicts`` (T8). The
function is byte-equivalent to the prior implementation: each result
row gains a resolved ``img_url`` (via ``kuaa.library.keyframe_url``)
and, when ``meta_by_scene`` is supplied, a SMPTE ``timecode`` field
computed from ``start_time_s`` (via ``kuaa.library.to_smpte``).

Both helpers moved under ``kuaa.library`` in P2/T4 — the prior
``api.services.catalog`` carve-out was deleted in T7.
"""

from __future__ import annotations

from pathlib import Path

from kuaa.library import keyframe_url, to_smpte


def results_to_dicts(
    results_df,
    data_dir: Path,
    meta_by_scene: dict | None = None,
    fps: float = 24.0,
) -> list[dict]:
    """Convert a search result DataFrame to the template's card dicts.

    When ``meta_by_scene`` is supplied (a ``{scene_id: kf_entry}`` dict
    from ``keyframes_metadata.json``), each result row is enriched with
    a SMPTE ``timecode`` field computed from ``start_time_s``. Without
    it the behaviour is byte-equivalent to the prior route
    implementation.
    """
    out = []
    for row in results_df.to_dict("records"):
        d = {**row, "img_url": keyframe_url(str(row["filepath"]), data_dir)}
        if meta_by_scene is not None:
            meta = meta_by_scene.get(row.get("scene_id"))
            if meta:
                start_s = float(meta.get("start_time_s") or 0.0)
                d["timecode"] = to_smpte(start_s, fps) if start_s > 0 else ""
        out.append(d)
    return out
