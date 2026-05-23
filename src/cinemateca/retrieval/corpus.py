"""Build a per-film BM25 corpus from descriptions + tag index.

Output shape: ``list[(scene_id, tokens)]``. Order is insertion-order of
the union of scene_ids found in descriptions ∪ tag_index, sorted
ascending by scene_id so reproducible across runs.

Scenes that have neither a description nor any tag do not appear in
the corpus. The BM25 index handles missing docs by simply not ranking
them — for those scenes only CLIP can rank, and the hybrid fusion
treats them as rank = "absent" → contributes 0 to the BM25 term.
"""

from __future__ import annotations

from collections.abc import Sequence

from cinemateca.retrieval.tokenize import tokenize


def build_corpus(
    descriptions: Sequence[dict],
    tag_index: dict[str, Sequence[int]],
    *,
    stopwords_lang: str | None = None,
) -> list[tuple[int, list[str]]]:
    """Build ``[(scene_id, tokens), …]`` for a single film.

    Args:
        descriptions: List of ``{"scene_id": int, "description": str}``
            dicts as read from ``scene_descriptions.json``.
        tag_index: Merged ``{tag: [scene_id, …]}`` mapping from
            ``load_tag_index(metadata_dir)``.
        stopwords_lang: Forwarded to ``tokenize``.

    Returns:
        Sorted-by-scene_id list of ``(scene_id, tokens)``. Scenes with
        neither description text nor any tag are omitted.
    """
    desc_by_sid: dict[int, str] = {}
    for entry in descriptions:
        sid = entry.get("scene_id")
        if sid is None:
            continue
        try:
            sid_int = int(sid)
        except (TypeError, ValueError):
            continue
        desc_by_sid[sid_int] = str(entry.get("description") or "")

    tags_by_sid: dict[int, list[str]] = {}
    for tag, sids in tag_index.items():
        if not isinstance(sids, (list, tuple, set)):
            continue
        for sid in sids:
            try:
                sid_int = int(sid)
            except (TypeError, ValueError):
                continue
            tags_by_sid.setdefault(sid_int, []).append(str(tag))

    all_sids = sorted(set(desc_by_sid) | set(tags_by_sid))
    docs: list[tuple[int, list[str]]] = []
    for sid in all_sids:
        desc = desc_by_sid.get(sid, "")
        tags = tags_by_sid.get(sid, [])
        # Tag list is concatenated to the description text; flat token
        # weighting (each tag = one token). Future tuning lever is a
        # tag_boost multiplier — see spec Risks #7.
        text = (desc + " " + " ".join(tags)).strip() if (desc or tags) else ""
        tokens = tokenize(text, stopwords_lang=stopwords_lang)
        if not tokens:
            continue
        docs.append((sid, tokens))
    return docs
