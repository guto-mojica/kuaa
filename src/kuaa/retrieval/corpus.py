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
from collections.abc import Mapping, Sequence

from kuaa.errors import is_error_response
from kuaa.retrieval.bilingual import expand_text
from kuaa.retrieval.tokenize import RegexTokenizer, Tokenizer

logger = logging.getLogger(__name__)


def build_corpus(
    descriptions: Sequence[dict],
    tag_index: Mapping[str, Sequence[int]],
    *,
    stopwords_lang: str | None = None,
    tokenizer: Tokenizer | None = None,
    tag_boost: int = 1,
    bilingual: bool = False,
) -> list[tuple[int, list[str]]]:
    """Build ``[(scene_id, tokens), …]`` for a single film.

    Args:
        descriptions: List of ``{"scene_id": int, "description": str}``
            dicts as read from ``scene_descriptions.json``.
        tag_index: Merged ``{tag: [scene_id, …]}`` mapping from
            ``load_tag_index(metadata_dir)``.
        stopwords_lang: Forwarded to the default ``RegexTokenizer`` when
            ``tokenizer`` is ``None``. Ignored if an explicit ``tokenizer``
            is provided.
        tokenizer: Optional :class:`~kuaa.retrieval.tokenize.Tokenizer`
            instance. When ``None``, a ``RegexTokenizer(stopwords_lang=…)`` is
            constructed from the ``stopwords_lang`` argument (preserving the
            legacy calling convention byte-for-byte).
        tag_boost: Per-surface weight on the *tags* surface. Tag tokens are
            repeated ``tag_boost`` times in each doc so a tag match carries
            more BM25 term-frequency than the same word in a description.
            ``1`` (the default) is byte-identical to flat concatenation —
            the curated-tag surface is cleaned by the Phase-1 suppression
            layer, and this lever lifts what remains. Values < 1 clamp to 1.
        bilingual: When True, append the Portuguese surface forms implied
            by each document's English content (see
            :mod:`kuaa.retrieval.bilingual`). The describer writes English
            captions and tags, so without this a Portuguese query has
            nothing to match. Additive — English recall is unchanged.

    Returns:
        Sorted-by-scene_id list of ``(scene_id, tokens)``. Scenes with
        no description text or tag are omitted.
    """
    # Resolve tokenizer: explicit injection wins; fall back to the legacy
    # RegexTokenizer path so existing callers are unaffected.
    _tok: Tokenizer = (
        tokenizer if tokenizer is not None else RegexTokenizer(stopwords_lang=stopwords_lang)
    )
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
    dropped_error_tags: set[str] = set()
    for tag, sids in tag_index.items():
        if not isinstance(sids, (list, tuple, set)):
            continue
        # Defence in depth for artefacts written before the describer learned
        # to reject captured failures: a swallowed backend exception could be
        # kebab-cased into the tag vocabulary and indexed as scene content.
        # ``the-great-train-robbery-1903`` still carries
        # ``error:-passed-cpu-tensor-to-mps-op`` on every one of its scenes.
        if is_error_response(str(tag).replace("-", " ")):
            dropped_error_tags.add(str(tag))
            continue
        for sid in sids:
            try:
                sid_int = int(sid)
            except (TypeError, ValueError):
                continue
            tags_by_sid.setdefault(sid_int, []).append(str(tag))
    if dropped_error_tags:
        logger.warning(
            "build_corpus: dropped %d error-shaped tag(s) from the index: %s — "
            "the film's describer run failed and wrote the failure as content; "
            "re-run the llm step for it",
            len(dropped_error_tags),
            sorted(dropped_error_tags),
        )

    all_sids = sorted(set(desc_by_sid) | set(tags_by_sid))
    docs: list[tuple[int, list[str]]] = []
    pruned_by_stopwords = 0
    for sid in all_sids:
        desc = desc_by_sid.get(sid, "")
        tags = tags_by_sid.get(sid, [])
        # Tokenise the description and tag surfaces independently — via the
        # resolved pluggable tokenizer (C6) — so the tag surface can be
        # weighted (repeated ``boost`` times). With boost == 1 the token
        # multiset/order matches flat concatenation (whitespace is a token
        # boundary either way), so the default path is byte-identical to the
        # pre-lever corpus.
        desc_tokens = _tok.tokenize(desc) if desc else []
        tags_text = " ".join(tags)
        tag_tokens = _tok.tokenize(tags_text) if tags_text else []
        tokens = desc_tokens + tag_tokens * boost
        text = " ".join(part for part in (desc, tags_text) if part).strip()
        if bilingual and text:
            # Append the PT surface forms implied by the English content, so a
            # Portuguese query has Portuguese to match. Routed through the same
            # tokenizer as everything else, so folding/stemming apply uniformly.
            # Strictly additive: no English token is removed, so no English
            # query loses a hit.
            pt_tokens = _tok.tokenize(expand_text(text))
            tokens = tokens + pt_tokens
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
