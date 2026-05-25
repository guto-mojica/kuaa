"""DataFrame → template-dict conversion. Private to the search package.

Extracted from ``api/services/search.py::results_to_dicts`` (T8). The
function is byte-equivalent to the prior implementation: each result
row gains a resolved ``img_url`` (via ``api.services.catalog.keyframe_url``)
and, when ``meta_by_scene`` is supplied, a SMPTE ``timecode`` field
computed from ``start_time_s`` (via ``api.services.catalog.to_smpte``).

The ``api.services.catalog`` imports cross the ``cinemateca → api``
boundary that ``no-core-imports-api`` normally forbids. The import is
carved out in ``.importlinter`` because both helpers will move under
``cinemateca.library`` in P2; the carve-out deletes then.
"""

from __future__ import annotations

from pathlib import Path

from api.services.catalog import keyframe_url, to_smpte


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
