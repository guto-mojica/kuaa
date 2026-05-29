"""Build a per-film BM25 corpus from descriptions + tag index.

Output shape: ``list[(scene_id, tokens)]``. Order is insertion-order of
the union of scene_ids found in descriptions ∪ tag_index, sorted
ascending by scene_id so reproducible across runs.

Scenes that have no description or tag do not appear in
the corpus. The BM25 index handles missing docs by simply not ranking
them — for those scenes only CLIP can rank, and the hybrid fusion
treats them as rank = "absent" → contributes 0 to the BM25 term.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from cinemateca.retrieval.tokenize import tokenize

logger = logging.getLogger(__name__)


def build_corpus(
    descriptions: Sequence[dict],
    tag_index: dict[str, Sequence[int]],
    *,
    stopwords_lang: str | None = None,
    tag_boost: int = 1,
) -> list[tuple[int, list[str]]]:
    """Build ``[(scene_id, tokens), …]`` for a single film.

    Args:
        descriptions: List of ``{"scene_id": int, "description": str}``
            dicts as read from ``scene_descriptions.json``.
        tag_index: Merged ``{tag: [scene_id, …]}`` mapping from
            ``load_tag_index(metadata_dir)``.
        stopwords_lang: Forwarded to ``tokenize``.
        tag_boost: Per-surface weight on the *tags* surface. Tag tokens are
            repeated ``tag_boost`` times in each doc so a tag match carries
            more BM25 term-frequency than the same word in a description.
            ``1`` (the default) is byte-identical to flat concatenation —
            the curated-tag surface is cleaned by the Phase-1 suppression
            layer, and this lever lifts what remains. Values < 1 clamp to 1.

    Returns:
        Sorted-by-scene_id list of ``(scene_id, tokens)``. Scenes with
        no description text or tag are omitted.
    """
    boost = max(1, int(tag_boost))
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
    pruned_by_stopwords = 0
    for sid in all_sids:
        desc = desc_by_sid.get(sid, "")
        tags = tags_by_sid.get(sid, [])
        # Tokenise the description and tag surfaces independently so the tag
        # surface can be weighted (repeated ``boost`` times). With boost == 1
        # the result is the same token multiset and order that flat
        # concatenation produced (whitespace is a token boundary either way),
        # so the default path is byte-identical to the pre-lever corpus.
        desc_tokens = tokenize(desc, stopwords_lang=stopwords_lang) if desc else []
        tags_text = " ".join(tags)
        tag_tokens = tokenize(tags_text, stopwords_lang=stopwords_lang) if tags_text else []
        tokens = desc_tokens + tag_tokens * boost
        text = " ".join(part for part in (desc, tags_text) if part).strip()
        if not tokens:
            # Distinguish two empty-token shapes:
            #   * text itself was empty (no desc + no tags) — silently
            #     skipped, the docstring documents this is expected.
            #   * text was NON-EMPTY but the tokenizer threw everything
            #     away (stopword over-pruning, pure-punctuation rows).
            #     This is a silent recall loss for hybrid mode — emit a
            #     debug log so operators tuning stopwords can see how
            #     much corpus they're losing.
            if text:
                pruned_by_stopwords += 1
                logger.debug(
                    "build_corpus: scene_id=%s text=%r tokenized to empty "
                    "(stopwords_lang=%s); not indexed",
                    sid,
                    text,
                    stopwords_lang,
                )
            continue
        docs.append((sid, tokens))
    if pruned_by_stopwords:
        logger.info(
            "build_corpus: %d scene(s) had non-empty text but tokenized "
            "to empty under stopwords_lang=%s — those scenes are "
            "BM25-invisible (will rank via CLIP only in hybrid mode)",
            pruned_by_stopwords,
            stopwords_lang,
        )
    return docs
