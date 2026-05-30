"""Per-film metadata loaders — read keyframes / descriptions / tags / visual JSON.

Deliberate divergence in this module: :func:`load_tag_index` is LENIENT on
malformed ``scene_tags.json`` (logs + returns ``{}``) while
:func:`load_metadata` reads the same file via :func:`load_json` STRICTLY
(propagates ``json.JSONDecodeError``). The asymmetry is inherited from P1
(``cinemateca.search._tag_index`` was lenient; the catalog twin
was strict) and preserved here on purpose:

  * Search-tab paths (which call :func:`load_tag_index`) degrade gracefully
    on a corrupted tag file — the search still returns CLIP-only results.
  * Scenes-tab paths (which call :func:`load_metadata`) surface the error
    immediately because the corruption blocks scene rendering anyway.

When P3+ unifies the two paths, pick one behavior across the module.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from cinemateca.library.paths import load_json
from cinemateca.scene_ids import scene_id_key

logger = logging.getLogger(__name__)


def load_tag_index(metadata_dir: Path) -> dict:
    """Load the RAW merged (un-normalized) inverted tag index.

    Reads ``scene_tags.json`` (LLM, INT ids) + ``manual_annotations.json``
    (STR keys) and ``merge_tag_index`` them WITHOUT normalizing.

    Tolerates malformed ``scene_tags.json`` (logs + treats as empty).
    """
    from cinemateca.annotations import load as load_annotations
    from cinemateca.annotations import merge_tag_index
    from cinemateca.annotations.overrides import load as load_overrides

    tags_path = metadata_dir / "scene_tags.json"
    llm_tags: dict = {}
    if tags_path.exists():
        try:
            with open(tags_path, encoding="utf-8") as f:
                llm_tags = json.load(f)
        except json.JSONDecodeError:
            logger.warning("load_tag_index: malformed %s; using empty tag index", tags_path)
            llm_tags = {}
    annotations = load_annotations(metadata_dir)
    overrides = load_overrides(metadata_dir)
    return merge_tag_index(llm_tags, annotations, overrides)


def load_metadata(
    metadata_dir: Path,
) -> tuple[list[Any], dict[Any, Any], dict[Any, Any], dict[Any, Any]]:
    """Return ``(kf_meta, desc_by_scene, vis_by_scene, tag_index)``.

    ``tag_index`` is NORMALIZED via ``normalize_tag_index``; ``desc_by_scene``
    and ``vis_by_scene`` are keyed by canonical str ids.
    """
    from cinemateca.annotations import load as load_annotations
    from cinemateca.annotations import merge_tag_index
    from cinemateca.annotations.overrides import load as load_overrides
    from cinemateca.scene_ids import normalize_tag_index

    raw_kf = load_json(metadata_dir / "keyframes_metadata.json")
    kf_meta: list[Any] = raw_kf if isinstance(raw_kf, list) else []
    raw_desc = load_json(metadata_dir / "scene_descriptions.json")
    descriptions: list[Any] = raw_desc if isinstance(raw_desc, list) else []
    raw_tags = load_json(metadata_dir / "scene_tags.json")
    llm_tags: dict[str, list[str]] | None = raw_tags if isinstance(raw_tags, dict) else None
    raw_vis = load_json(metadata_dir / "visual_analysis.json")
    visual_data: list[Any] = raw_vis if isinstance(raw_vis, list) else []
    annotations = load_annotations(metadata_dir)
    overrides = load_overrides(metadata_dir)

    desc_by_scene = {scene_id_key(d["scene_id"]): d for d in descriptions if "scene_id" in d}
    vis_by_scene = {scene_id_key(v["scene_id"]): v for v in visual_data if "scene_id" in v}
    tag_index = normalize_tag_index(merge_tag_index(llm_tags, annotations, overrides))

    return kf_meta, desc_by_scene, vis_by_scene, tag_index
