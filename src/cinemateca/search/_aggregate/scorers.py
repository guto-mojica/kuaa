"""Per-film scorers for the decomposed aggregate pipeline (C1).

Each scorer takes one film's loaded artefacts and returns a per-film
ranked ``list[(scene_id, score)]``. Bodies are moved verbatim from the
pre-C1 ``aggregate_search`` so behavior is byte-identical (snapshot-gated).
"""

from __future__ import annotations

import re
from typing import Any

from cinemateca.retrieval.tokenize import tokenize

_SCENE_ID_FROM_PATH_RE = re.compile(r"Scene-(\d+)", flags=re.IGNORECASE)


def _scene_id_from_visual_record(record: dict[str, Any]) -> int | None:
    """Best-effort scene id extraction for visual-analysis rows."""
    sid = record.get("scene_id")
    if sid is not None:
        try:
            return int(sid)
        except (TypeError, ValueError):
            return None
    frame_path = str(record.get("frame_path") or record.get("filepath") or "")
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


def _phrase_match_score(
    value: Any, query_tokens: list[str], *, exact: float, contains: float
) -> float:
    tokens = _tokens_for_value(value)
    if not tokens or not query_tokens:
        return 0.0
    q = tuple(query_tokens)
    if tuple(tokens) == q:
        return exact
    token_set = set(tokens)
    if all(t in token_set for t in query_tokens):
        return contains
    return 0.0


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
    ) -> dict[int, float]:
        """Return exact metadata/object match scores keyed by scene id."""
        query_tokens = tokenize(query)
        if not query_tokens or len(query_tokens) > 4:
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

        for tag, sids in tag_index.items():
            tag_score = _phrase_match_score(tag, query_tokens, exact=0.25, contains=0.1)
            if tag_score <= 0 or not isinstance(sids, (list, tuple, set)):
                continue
            for sid in sids:
                add(sid, tag_score)

        for entry in descriptions:
            sid = entry.get("scene_id")
            if sid is None:
                continue
            desc_score = _phrase_match_score(
                entry.get("description"), query_tokens, exact=12.0, contains=12.0
            )
            action_score = _phrase_match_score(
                entry.get("people_action"), query_tokens, exact=2.0, contains=2.0
            )
            add(sid, desc_score)
            add(sid, action_score)
            has_description_evidence = desc_score > 0.0

            # Structured generated labels support the written description/action.
            # Alone, they are intentionally weak: these labels can contain loose
            # object guesses that are noisier than the prose or detector output.
            structured_exact = 3.0 if has_description_evidence else 1.0
            structured_contains = 2.0 if has_description_evidence else 0.5
            for key in ("objects", "tags"):
                add(
                    sid,
                    _phrase_match_score(
                        entry.get(key),
                        query_tokens,
                        exact=structured_exact,
                        contains=structured_contains,
                    ),
                )
            for key in ("setting", "location"):
                add(sid, _phrase_match_score(entry.get(key), query_tokens, exact=2.0, contains=1.0))
            raw = entry.get("_raw_responses")
            if isinstance(raw, dict):
                add(
                    sid,
                    _phrase_match_score(
                        raw.get("objects"),
                        query_tokens,
                        exact=structured_exact,
                        contains=structured_contains,
                    ),
                )

        for row in visual_rows:
            sid = _scene_id_from_visual_record(row)
            if sid is None:
                continue
            obj = row.get("object_detection")
            if not isinstance(obj, dict):
                continue
            for detected in obj.get("objects") or []:
                if isinstance(detected, dict):
                    add(
                        sid,
                        _phrase_match_score(
                            detected.get("class"), query_tokens, exact=10.0, contains=7.0
                        ),
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
                            n * _phrase_match_score(cls, query_tokens, exact=10.0, contains=7.0),
                        ),
                    )

        return scores


class CLIPScorer:
    """Per-film CLIP cosine ranker (best keyframe per scene). Step D fills this in."""


class BM25Scorer:
    """Per-film BM25 ranker over the loaded corpus. Step D fills this in."""


__all__ = [
    "MetadataScorer",
    "CLIPScorer",
    "BM25Scorer",
    "_scene_id_from_visual_record",
    "_tokens_for_value",
    "_phrase_match_score",
]
