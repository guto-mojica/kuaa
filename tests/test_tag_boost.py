"""Tests for the per-surface BM25 ``tag_boost`` lever (Phase 2).

``tag_boost=1`` (the default) must be byte-identical to the pre-lever flat
concatenation; ``tag_boost>1`` repeats tag tokens so tag matches carry more
BM25 term-frequency. The loader cache must key on ``tag_boost`` too.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from kuaa.retrieval.corpus import build_corpus
from kuaa.retrieval.tokenize import tokenize


def test_tag_boost_default_is_byte_identical() -> None:
    descriptions = [{"scene_id": 1, "description": "a man walking outdoors"}]
    tag_index = {"horse": [1], "rural": [1]}

    got = build_corpus(descriptions, tag_index)
    explicit_one = build_corpus(descriptions, tag_index, tag_boost=1)
    # Reconstruct the pre-lever flat-concatenation tokens.
    flat = tokenize("a man walking outdoors horse rural")

    assert got == explicit_one == [(1, flat)]


def test_tag_boost_repeats_only_tag_tokens() -> None:
    descriptions = [{"scene_id": 1, "description": "man"}]
    tag_index = {"horse": [1]}

    one = build_corpus(descriptions, tag_index, tag_boost=1)[0][1]
    three = build_corpus(descriptions, tag_index, tag_boost=3)[0][1]

    assert one == ["man", "horse"]
    # Description token kept once; tag token repeated tag_boost times.
    assert three == ["man", "horse", "horse", "horse"]
    assert three.count("man") == 1


def test_tag_boost_clamps_below_one() -> None:
    docs = build_corpus([{"scene_id": 1, "description": "man"}], {"horse": [1]}, tag_boost=0)
    assert docs == [(1, ["man", "horse"])]


def test_tag_boost_lifts_tag_match_ranking() -> None:
    """A scene that matches the query only via a tag should out-rank a scene
    that matches only via description once the tag surface is boosted."""
    from kuaa.retrieval.bm25 import BM25Index

    # Scene 1 mentions "boat" in its description; scene 2 carries it as a tag.
    descriptions = [
        {"scene_id": 1, "description": "boat on the wide river at dawn light"},
        {"scene_id": 2, "description": "a long quiet afternoon scene indoors"},
        {"scene_id": 3, "description": "unrelated street traffic at night"},
    ]
    tag_index = {"boat": [2]}

    flat = BM25Index.build(descriptions=descriptions, tag_index=tag_index, tag_boost=1)
    boosted = BM25Index.build(descriptions=descriptions, tag_index=tag_index, tag_boost=8)

    flat_score = dict(flat.query("boat", top_k=5))
    boosted_score = dict(boosted.query("boat", top_k=5))

    # Boosting raises scene 2's tag-only match relative to its flat score.
    assert boosted_score[2] > flat_score[2]


def test_loader_cache_keys_on_tag_boost(tmp_path: Path) -> None:
    from kuaa.search.bm25 import bm25_index_for_dir

    md = tmp_path / "metadata"
    md.mkdir()
    (md / "scene_descriptions.json").write_text(
        json.dumps(
            [
                {"scene_id": 0, "description": "barco no rio"},
                {"scene_id": 1, "description": "homem na rua"},
                {"scene_id": 2, "description": "carro na estrada"},
            ]
        )
    )
    (md / "scene_tags.json").write_text(json.dumps({"barco": [0]}))

    a = bm25_index_for_dir(metadata_dir=md, stopwords_lang=None, k1=1.5, b=0.75, tag_boost=1)
    a2 = bm25_index_for_dir(metadata_dir=md, stopwords_lang=None, k1=1.5, b=0.75, tag_boost=1)
    assert a is a2, "same tag_boost must hit the cache"

    time.sleep(0.001)
    b = bm25_index_for_dir(metadata_dir=md, stopwords_lang=None, k1=1.5, b=0.75, tag_boost=3)
    assert b is not a, "different tag_boost must produce a distinct cache entry"
