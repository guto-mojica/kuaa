"""Unit tests for per-film corpus builder.

The corpus builder is pure-functional: given a descriptions list + a
merged tag-index, it returns a list of ``(scene_id, tokens)`` docs
ready for BM25 indexing.
"""

from __future__ import annotations

import pytest

from kuaa.retrieval.corpus import build_corpus


def _descriptions(items: list[tuple[int, str]]) -> list[dict]:
    """Helper: shape descriptions list the way scene_descriptions.json does."""
    return [{"scene_id": sid, "description": desc} for sid, desc in items]


def test_builds_docs_from_descriptions_and_tags() -> None:
    descs = _descriptions(
        [
            (0, "menina chorando na chuva"),
            (1, "homem caminhando"),
            (2, "casa abandonada"),
        ]
    )
    tag_index = {
        "exterior": [0, 1],
        "interior": [2],
        "noite": [0],
    }
    docs = build_corpus(descs, tag_index)
    by_sid = dict(docs)
    assert set(by_sid.keys()) == {0, 1, 2}
    assert "menina" in by_sid[0]
    assert "exterior" in by_sid[0]
    assert "noite" in by_sid[0]
    assert "homem" in by_sid[1]
    assert "exterior" in by_sid[1]
    assert "casa" in by_sid[2]
    assert "interior" in by_sid[2]


def test_missing_description_uses_tags_only() -> None:
    docs = build_corpus(_descriptions([]), {"exterior": [0]})
    by_sid = dict(docs)
    assert 0 in by_sid
    assert "exterior" in by_sid[0]


def test_missing_tags_uses_description_only() -> None:
    docs = build_corpus(
        _descriptions([(0, "menina chorando")]),
        tag_index={},
    )
    by_sid = dict(docs)
    assert by_sid[0] == ["menina", "chorando"]


def test_both_missing_emits_no_doc_for_that_scene() -> None:
    docs = build_corpus(_descriptions([]), tag_index={})
    assert docs == []


def test_empty_description_string_falls_back_to_tags() -> None:
    docs = build_corpus(
        _descriptions([(0, "")]),
        tag_index={"exterior": [0]},
    )
    by_sid = dict(docs)
    assert by_sid[0] == ["exterior"]


def test_pass_through_stopwords_lang() -> None:
    pytest.importorskip("nltk")
    docs = build_corpus(
        _descriptions([(0, "filme de teste")]),
        tag_index={},
        stopwords_lang="pt",
    )
    by_sid = dict(docs)
    assert "de" not in by_sid[0]
