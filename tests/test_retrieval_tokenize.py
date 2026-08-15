"""Unit tests for the retrieval tokenizer.

The tokenizer is the entry point of the BM25 corpus pipeline. Keep it
pure (no I/O, no globals); every behavior here must be verifiable from
inputs alone.
"""

from __future__ import annotations

from kuaa.retrieval.tokenize import tokenize


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
    out = tokenize("filme de teste", stopwords_lang="pt")
    assert "de" not in out
    assert "filme" in out
    assert "teste" in out


def test_normalisation_is_opt_in() -> None:
    """Defaults stay lossless — accents and plurals survive untouched."""
    assert tokenize("única cavalos") == ["única", "cavalos"]


def test_fold_and_stem_flags() -> None:
    assert tokenize("única", fold=True) == ["unica"]
    assert tokenize("cavalos", fold=True, stem=True) == ["cavalo"]


def test_empty_string_returns_empty_list() -> None:
    assert tokenize("") == []
    assert tokenize("   ") == []
