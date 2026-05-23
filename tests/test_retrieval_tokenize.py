"""Unit tests for the retrieval tokenizer.

The tokenizer is the entry point of the BM25 corpus pipeline. Keep it
pure (no I/O, no globals); every behavior here must be verifiable from
inputs alone.
"""

from __future__ import annotations

import pytest

from cinemateca.retrieval.tokenize import tokenize


def test_lowercases_and_strips_punctuation() -> None:
    assert tokenize("Hello, World!") == ["hello", "world"]


def test_preserves_pt_diacritics() -> None:
    assert tokenize("São Paulo é uma cidade") == ["são", "paulo", "é", "uma", "cidade"]


def test_drops_single_character_tokens() -> None:
    assert tokenize("a casa e o cão") == ["casa", "cão"]


def test_handles_numbers() -> None:
    assert tokenize("filme de 1959") == ["filme", "de", "1959"]


def test_stopwords_lang_none_keeps_all_tokens() -> None:
    assert "de" in tokenize("filme de teste", stopwords_lang=None)


def test_stopwords_lang_pt_removes_pt_stopwords() -> None:
    pytest.importorskip("nltk", reason="nltk optional; skip if not installed")
    out = tokenize("filme de teste", stopwords_lang="pt")
    assert "de" not in out
    assert "filme" in out
    assert "teste" in out


def test_empty_string_returns_empty_list() -> None:
    assert tokenize("") == []
    assert tokenize("   ") == []
