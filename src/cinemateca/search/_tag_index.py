"""Shared tag-index loader for the search package.

Promoted from ``cinemateca.search.bm25._load_tag_index`` (T7) so the
loader has one home and can be imported by every extracted search
module that needs it (T7 BM25 loader, T8 Mojica context builders, and
P2 helpers that move under ``cinemateca.library``).

Mirrors ``api.services.catalog.load_tag_index`` shape-for-shape:
reads ``scene_tags.json`` (LLM-side, INT scene_id keys) and merges
with manual annotations (STR scene_id keys) via
``cinemateca.annotator.merge_tag_index``. Returns the RAW
(un-normalised) merged inverted index.

Divergence from the catalog twin (intentional, observed only on the
unhappy path): malformed JSON in ``scene_tags.json`` is tolerated here
(logged + empty), matching the BM25 loader's resilience contract. The
catalog twin still raises on the same condition. Both contracts are
characterised by their own tests; this module preserves the BM25 one.

Lives here (rather than in ``api.services.catalog``) so the
``cinemateca`` package stays free of any ``api`` import — enforced by
import-linter's ``no-core-imports-api`` contract. P2 will move the
catalog twin under ``cinemateca.library``, at which point the two
contracts can be reconciled.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_tag_index(metadata_dir: Path) -> dict:
    """Load the RAW merged (un-normalised) inverted tag index.

    See module docstring for the divergence note vs the catalog twin.
    """
    from cinemateca.annotator import load as load_annotations
    from cinemateca.annotator import merge_tag_index

    tags_path = metadata_dir / "scene_tags.json"
    llm_tags: dict = {}
    if tags_path.exists():
        try:
            with open(tags_path, encoding="utf-8") as f:
                llm_tags = json.load(f)
        except json.JSONDecodeError:
            logger.warning("search: malformed %s; using empty tag index", tags_path)
            llm_tags = {}
    annotations = load_annotations(metadata_dir)
    return merge_tag_index(llm_tags, annotations)
