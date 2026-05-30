"""C6 — pluggable tokenizer; PT-aware option folds diacritics/stopwords."""

from __future__ import annotations

from cinemateca.retrieval.tokenize import (
    RegexTokenizer,
    Tokenizer,
    get_tokenizer,
)


def test_get_tokenizer_default_is_regex() -> None:
    tok = get_tokenizer("regex")
    assert isinstance(tok, RegexTokenizer)
    assert isinstance(tok, Tokenizer)  # runtime-checkable Protocol


def test_regex_tokenizer_matches_legacy_behavior() -> None:
    # length-1 ASCII dropped, non-ASCII length-1 kept (legacy contract).
    assert get_tokenizer("regex").tokenize("é a o cavalo") == ["é", "cavalo"]


def test_multilingual_tokenizer_strips_pt_stopwords() -> None:
    import pytest

    pytest.importorskip("nltk")
    toks = get_tokenizer("multilingual").tokenize("o homem no cavalo")
    # PT stopwords "o" / "no" removed; content words kept.
    assert "cavalo" in toks
    assert "o" not in toks


def test_unknown_tokenizer_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="tokenizer"):
        get_tokenizer("klingon")
