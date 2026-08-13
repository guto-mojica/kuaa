"""Per-film scorers for the decomposed aggregate pipeline (C1).

Each scorer takes one film's loaded artefacts and returns a per-film
ranked ``list[(scene_id, score)]``. Bodies are moved verbatim from the
pre-C1 ``aggregate_search`` so behavior is byte-identical (snapshot-gated).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kuaa.retrieval.bilingual import en_forms
from kuaa.retrieval.tokenize import tokenize
from kuaa.scene_ids import scene_id_key
from kuaa.search._aggregate.coco_aliases import with_coco_aliases

_SCENE_ID_FROM_PATH_RE = re.compile(r"Scene-(\d+)", flags=re.IGNORECASE)

# A bound ``MetadataScorer.score.add`` sink: ``add(scene_id, delta)`` folds a
# positive ``delta`` into the per-scene accumulator (no-op for delta <= 0 or
# an unparsable scene id). Each scoring pass below takes this sink so the
# three passes share one ``scores`` dict.
_AddFn = Callable[[Any, float], None]


def _scene_id_from_visual_record(
    record: dict[str, Any], frame_to_scene: dict[str, Any] | None = None
) -> int | None:
    """Best-effort scene id extraction for visual-analysis rows.

    Resolution order: an explicit ``scene_id`` on the record, then the
    ``keyframes_metadata.json`` manifest (``frame_to_scene``), then the legacy
    ``Scene-NNN`` filename regex. No record on disk actually carries a
    ``scene_id``, so the manifest is the leg that does the work; the regex is
    kept only for partial metadata dirs where the manifest is unavailable, and
    it matches just one of the two live naming conventions.
    """
    sid = record.get("scene_id")
    if sid is not None:
        try:
            return int(sid)
        except (TypeError, ValueError):
            return None
    frame_path = str(record.get("frame_path") or record.get("filepath") or "")
    if frame_to_scene:
        mapped = frame_to_scene.get(Path(frame_path).name)
        if mapped is not None:
            try:
                return int(mapped)
            except (TypeError, ValueError):
                return None
    match = _SCENE_ID_FROM_PATH_RE.search(frame_path)
    if match:
        return int(match.group(1))
    return None


def _tokens_for_value(value: Any) -> list[str]:
    """Tokenize nested metadata values without assuming a fixed schema."""
    if value is None:
        return []
    if isinstance(value, str):
        return tokenize(value)
    if isinstance(value, (int, float, bool)):
        return tokenize(str(value))
    if isinstance(value, (list, tuple, set)):
        item_tokens: list[str] = []
        for item in value:
            item_tokens.extend(_tokens_for_value(item))
        return item_tokens
    if isinstance(value, dict):
        dict_tokens: list[str] = []
        for item in value.values():
            dict_tokens.extend(_tokens_for_value(item))
        return dict_tokens
    return tokenize(str(value))


# Partial-coverage gates. A query only earns proportional credit once it
# matches at least this many terms AND this share of them — below that,
# a single incidental word ("with", "scene") would score every document.
_MIN_PARTIAL_TERMS = 2
_MIN_PARTIAL_COVERAGE = 0.5

# Cost bound on how many query terms are scored. Replaces the previous
# hard `len(query_tokens) > 4 -> return {}` bail-out, which silently
# disabled this entire leg — weighted 0.65, the largest share of hybrid
# fusion — on any natural-language query. Measured against the project's
# own eval slate, that guard fired on 13 of 15 text queries, which is why
# the `hybrid` and `hybrid-metadata` ablation rows were byte-identical.
_MAX_SCORED_TERMS = 8


@dataclass(frozen=True)
class _QueryTerms:
    """The surface forms that satisfy each term of a tokenised query.

    One variant set per token the user typed — the token itself plus its
    cross-language equivalents. Keeping the grouping (rather than a flat
    token list) means expansion widens what *counts* as a match without
    inflating the coverage denominator.
    """

    groups: tuple[frozenset[str], ...]

    def __bool__(self) -> bool:
        return bool(self.groups)

    def __len__(self) -> int:
        return len(self.groups)


def build_query_terms(query: str) -> _QueryTerms:
    """Tokenise ``query`` and attach each term's cross-language variants.

    Portuguese query terms gain their English equivalents because the
    metadata being matched is English: the describer prompts are English,
    so ``description`` / ``objects`` / ``setting`` come back in English,
    and YOLOv8 emits the 80 English COCO class names. Without this a
    curator searching ``cavalo`` scores nothing against a scene the
    system labelled ``horse``.

    Portuguese function words are dropped, because coverage is measured as
    a *share* of query terms: leaving ``com`` / ``ao`` / ``de`` in the
    denominator makes a well-matched query look poorly matched. Only the
    denominator changes — nothing stops a function word from matching.

    Truncated to :data:`_MAX_SCORED_TERMS` to bound cost on long queries.
    """
    tokens = tokenize(query, stopwords_lang="pt")[:_MAX_SCORED_TERMS]
    if not tokens:
        # An all-stopword query still deserves its literal shot.
        tokens = tokenize(query)[:_MAX_SCORED_TERMS]
    groups = tuple(
        # Two PT→EN sources, unioned: the general archive lexicon and the
        # narrower COCO class-name table the detector leg has always used.
        frozenset((token, *en_forms(token), *with_coco_aliases([token])))
        for token in tokens
    )
    return _QueryTerms(groups=groups)


def _phrase_match_score(value: Any, terms: _QueryTerms, *, exact: float, contains: float) -> float:
    """Score one metadata value against the query's terms.

    ``exact`` when the value is precisely the query, ``contains`` when
    every term is present, and proportional credit below that — gated by
    :data:`_MIN_PARTIAL_TERMS` / :data:`_MIN_PARTIAL_COVERAGE`.

    Proportional credit is what makes this leg usable on real queries.
    The previous all-or-nothing rule required *every* query token to be
    present, so a seven-word description of a shot scored zero against
    the very scene it described unless the phrasing matched exactly.
    """
    tokens = _tokens_for_value(value)
    if not tokens or not terms:
        return 0.0
    token_set = frozenset(tokens)
    matched = sum(1 for variants in terms.groups if token_set & variants)
    if matched == len(terms):
        # ``exact`` when the value carries nothing *beyond* the query terms.
        # Compared through the variant groups rather than against the typed
        # tokens, so a PT query earns the same exact-match tier against an
        # English detector label that the English query does — `cavalo` and
        # `horse` must score identically against the class `horse`.
        if len(token_set) == len(terms):
            return exact
        return contains
    if matched < _MIN_PARTIAL_TERMS or matched / len(terms) < _MIN_PARTIAL_COVERAGE:
        return 0.0
    return contains * (matched / len(terms))


def _score_tags(add: _AddFn, terms: _QueryTerms, tag_index: dict[str, Any]) -> None:
    """Fold exact/contains tag-name matches into the scene accumulator."""
    for tag, sids in tag_index.items():
        tag_score = _phrase_match_score(tag, terms, exact=0.25, contains=0.1)
        if tag_score <= 0 or not isinstance(sids, (list, tuple, set)):
            continue
        for sid in sids:
            add(sid, tag_score)


def _score_descriptions(
    add: _AddFn, terms: _QueryTerms, descriptions: list[dict[str, Any]]
) -> None:
    """Fold description / action / structured-label matches into the accumulator.

    The written ``description`` and ``people_action`` are the strong prose
    signal; structured generated labels (``objects`` / ``tags`` / raw-response
    objects) are intentionally weaker and weaker still when no prose match
    backs them, since those labels can carry loose object guesses.
    """
    for entry in descriptions:
        sid = entry.get("scene_id")
        if sid is None:
            continue
        desc_score = _phrase_match_score(entry.get("description"), terms, exact=12.0, contains=12.0)
        action_score = _phrase_match_score(
            entry.get("people_action"), terms, exact=2.0, contains=2.0
        )
        add(sid, desc_score)
        add(sid, action_score)
        has_description_evidence = desc_score > 0.0

        structured_exact = 3.0 if has_description_evidence else 1.0
        structured_contains = 2.0 if has_description_evidence else 0.5
        for key in ("objects", "tags"):
            add(
                sid,
                _phrase_match_score(
                    entry.get(key),
                    terms,
                    exact=structured_exact,
                    contains=structured_contains,
                ),
            )
        for key in ("setting", "location"):
            add(sid, _phrase_match_score(entry.get(key), terms, exact=2.0, contains=1.0))
        raw = entry.get("_raw_responses")
        if isinstance(raw, dict):
            add(
                sid,
                _phrase_match_score(
                    raw.get("objects"),
                    terms,
                    exact=structured_exact,
                    contains=structured_contains,
                ),
            )


def _score_visual_rows(
    add: _AddFn,
    terms: _QueryTerms,
    visual_rows: list[dict[str, Any]],
    frame_to_scene: dict[str, Any] | None = None,
) -> None:
    """Fold detector ``object_detection`` class hits into the accumulator.

    Per-object class matches add a flat weight; ``class_counts`` matches scale
    by the (clamped) detected count, capped at the description-tier ceiling.

    The detector emits English COCO labels while the interface language is
    pt-BR; the PT→EN bridge now lives in :func:`build_query_terms`, which
    applies it to every surface rather than to this leg alone.
    """
    for row in visual_rows:
        sid = _scene_id_from_visual_record(row, frame_to_scene)
        if sid is None:
            continue
        obj = row.get("object_detection")
        if not isinstance(obj, dict):
            continue
        for detected in obj.get("objects") or []:
            if isinstance(detected, dict):
                add(
                    sid,
                    _phrase_match_score(detected.get("class"), terms, exact=10.0, contains=7.0),
                )
        class_counts = obj.get("class_counts")
        if isinstance(class_counts, dict):
            for cls, count in class_counts.items():
                try:
                    n = max(1.0, float(count))
                except (TypeError, ValueError):
                    n = 1.0
                add(
                    sid,
                    min(
                        12.0,
                        n * _phrase_match_score(cls, terms, exact=10.0, contains=7.0),
                    ),
                )


class MetadataScorer:
    """Lexical exact-match scorer over tags / descriptions / detected objects.

    This is intentionally lexical. SigLIP handles broad semantic similarity;
    this signal protects short object queries such as ``dog`` where exact tags,
    visual object classes, and description/object fields are stronger evidence
    than a weak visual cosine rank.
    """

    def score(
        self,
        *,
        query: str,
        descriptions: list[dict[str, Any]],
        tag_index: dict[str, Any],
        visual_rows: list[dict[str, Any]],
        frame_to_scene: dict[str, Any] | None = None,
    ) -> dict[int, float]:
        """Return exact metadata/object match scores keyed by scene id.

        ``frame_to_scene`` is the ``keyframes_metadata.json`` frame → scene
        manifest (see :func:`kuaa.library.frame_to_scene_index`). Without it the
        visual-analysis rows cannot be attributed to a scene for most films and
        the object leg of this scorer contributes nothing.
        """
        terms = build_query_terms(query)
        if not terms:
            return {}

        scores: dict[int, float] = {}

        def add(sid: Any, delta: float) -> None:
            if delta <= 0:
                return
            try:
                sid_int = int(sid)
            except (TypeError, ValueError):
                return
            scores[sid_int] = scores.get(sid_int, 0.0) + delta

        _score_tags(add, terms, tag_index)
        _score_descriptions(add, terms, descriptions)
        _score_visual_rows(add, terms, visual_rows, frame_to_scene)
        return scores


class CLIPScorer:
    """Per-film CLIP cosine ranker — best keyframe per scene, descending."""

    def score(
        self,
        *,
        embeddings: Any,
        kf_df: Any,
        text_vec: np.ndarray,
        min_similarity: float,
        allowed_scene_keys: set[str] | None,
        raw_k: int,
    ) -> tuple[list[tuple[int, float]], dict[int, int]]:
        """Return ``(clip_ranked, best_row_by_sid)`` for one film.

        ``clip_ranked`` is ``[(scene_id, cosine)]`` sorted descending and
        truncated to ``raw_k``; ``best_row_by_sid`` maps each surfaced
        scene_id to the kf_df row index with the highest cosine (so the
        materialised keyframe points at the actual best-matching frame).
        """
        scores: np.ndarray = embeddings @ text_vec

        # CLIP-side ranked list — `(scene_id, cosine_score)` descending.
        # Best-keyframe-per-scene: a single scene may have multiple
        # keyframes (Phase-1 density), so the same scene_id can appear N
        # times in ``scores`` at different rows. Keep the row index with
        # the HIGHEST cosine per scene_id so the surfaced
        # ``keyframe_path`` points at the actual best-matching frame.
        best_score_by_sid: dict[int, float] = {}
        best_row_by_sid: dict[int, int] = {}
        for i, score in enumerate(scores):
            s = float(score)
            if s < min_similarity:
                continue
            row = kf_df.iloc[i]
            sid = int(row["scene_id"])
            if allowed_scene_keys is not None and scene_id_key(sid) not in allowed_scene_keys:
                continue
            prev = best_score_by_sid.get(sid)
            if prev is None or s > prev:
                best_score_by_sid[sid] = s
                best_row_by_sid[sid] = i
        clip_ranked: list[tuple[int, float]] = sorted(
            best_score_by_sid.items(), key=lambda p: p[1], reverse=True
        )[:raw_k]
        return clip_ranked, best_row_by_sid


class BM25Scorer:
    """Per-film BM25 ranker over a pre-loaded corpus index."""

    def score(
        self,
        *,
        bm25: Any,
        query: str,
        raw_k: int,
        allowed_scene_keys: set[str] | None,
    ) -> list[tuple[int, float]]:
        """Return ``[(scene_id, bm25_score)]`` for one film.

        ``bm25`` is the pre-loaded :class:`BM25Index` (or ``None``). A
        ``None`` index or one whose ``model`` is unbuilt contributes no
        entries (empty list) — the legacy fallback-on-empty contract,
        preserved verbatim. When ``allowed_scene_keys`` is set, hits are
        filtered to that tag-intersected scene set.
        """
        bm25_hits: list[tuple[int, float]] = []
        if bm25 is None or bm25.model is None:
            return bm25_hits
        bm25_hits = bm25.query(query, top_k=raw_k)
        if allowed_scene_keys is not None:
            bm25_hits = [
                (sid, s) for sid, s in bm25_hits if scene_id_key(sid) in allowed_scene_keys
            ]
        return bm25_hits


__all__ = [
    "MetadataScorer",
    "CLIPScorer",
    "BM25Scorer",
    "_scene_id_from_visual_record",
    "_tokens_for_value",
    "_phrase_match_score",
    "_score_tags",
    "_score_descriptions",
    "_score_visual_rows",
]
