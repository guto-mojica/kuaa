"""Build a per-film BM25 corpus from descriptions + transcripts + tag index.

Output shape: ``list[(scene_id, tokens)]``. Order is insertion-order of
the union of scene_ids found in descriptions ∪ transcripts ∪ tag_index, sorted
ascending by scene_id so reproducible across runs.

Scenes that have no description, transcript, or tag do not appear in
the corpus. The BM25 index handles missing docs by simply not ranking
them — for those scenes only CLIP can rank, and the hybrid fusion
treats them as rank = "absent" → contributes 0 to the BM25 term.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

from cinemateca.retrieval.tokenize import RegexTokenizer, Tokenizer

logger = logging.getLogger(__name__)


def build_corpus(
    descriptions: Sequence[dict],
    tag_index: Mapping[str, Sequence[int]],
    *,
    transcripts: Sequence[dict] | None = None,
    stopwords_lang: str | None = None,
    tokenizer: Tokenizer | None = None,
) -> list[tuple[int, list[str]]]:
    """Build ``[(scene_id, tokens), …]`` for a single film.

    Args:
        descriptions: List of ``{"scene_id": int, "description": str}``
            dicts as read from ``scene_descriptions.json``.
        tag_index: Merged ``{tag: [scene_id, …]}`` mapping from
            ``load_tag_index(metadata_dir)``.
        transcripts: Optional list of ``{"scene_id": int, "text": str}``
            dicts as read from ``scene_transcripts.json``.
        stopwords_lang: Forwarded to the default ``RegexTokenizer`` when
            ``tokenizer`` is ``None``. Ignored if an explicit ``tokenizer``
            is provided.
        tokenizer: Optional :class:`~cinemateca.retrieval.tokenize.Tokenizer`
            instance. When ``None``, a ``RegexTokenizer(stopwords_lang=…)`` is
            constructed from the ``stopwords_lang`` argument (preserving the
            legacy calling convention byte-for-byte).

    Returns:
        Sorted-by-scene_id list of ``(scene_id, tokens)``. Scenes with
        no description text, transcript text, or tag are omitted.
    """
    # Resolve tokenizer: explicit injection wins; fall back to the legacy
    # RegexTokenizer path so existing callers are unaffected.
    _tok: Tokenizer = (
        tokenizer if tokenizer is not None else RegexTokenizer(stopwords_lang=stopwords_lang)
    )
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

    transcript_by_sid: dict[int, str] = {}
    for entry in transcripts or ():
        sid = entry.get("scene_id")
        if sid is None:
            continue
        try:
            sid_int = int(sid)
        except (TypeError, ValueError):
            continue
        transcript_by_sid[sid_int] = str(entry.get("text") or "")

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

    all_sids = sorted(set(desc_by_sid) | set(transcript_by_sid) | set(tags_by_sid))
    docs: list[tuple[int, list[str]]] = []
    pruned_by_stopwords = 0
    for sid in all_sids:
        desc = desc_by_sid.get(sid, "")
        transcript = transcript_by_sid.get(sid, "")
        tags = tags_by_sid.get(sid, [])
        # Text surfaces are concatenated with flat token weighting.
        # Future tuning levers are per-surface boosts (tag_boost,
        # transcript_boost) once a larger corpus gives stable metrics.
        text = " ".join(part for part in (desc, transcript, " ".join(tags)) if part).strip()
        tokens = _tok.tokenize(text)
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
                    "(tokenizer=%s); not indexed",
                    sid,
                    text,
                    type(_tok).__name__,
                )
            continue
        docs.append((sid, tokens))
    if pruned_by_stopwords:
        logger.info(
            "build_corpus: %d scene(s) had non-empty text but tokenized "
            "to empty under tokenizer=%s — those scenes are "
            "BM25-invisible (will rank via CLIP only in hybrid mode)",
            pruned_by_stopwords,
            type(_tok).__name__,
        )
    return docs
