"""C6 — pluggable tokenizer; PT-aware option folds diacritics/stopwords."""

from __future__ import annotations

from kuaa.retrieval.tokenize import (
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
    # Stopwords are inlined, not loaded from nltk — no optional dep, and no
    # ModuleNotFoundError escaping the corpus build.
    toks = get_tokenizer("multilingual").tokenize("o homem no cavalo")
    # PT stopwords "o" / "no" removed; content words kept.
    assert "cavalo" in toks
    assert "o" not in toks


def test_multilingual_tokenizer_keeps_absence_and_negation_words() -> None:
    """``sem`` is load-bearing: the describer writes the tag ``sem-pessoas``.

    A full nltk PT stopword list drops it, which would make the system
    unable to match a tag it generated itself.
    """
    toks = get_tokenizer("multilingual").tokenize("cena sem pessoas e nao vazia")
    assert "sem" in toks
    assert "nao" in toks


def test_multilingual_tokenizer_folds_diacritics() -> None:
    # The pipeline writes `pessoa-unica`; a curator types `única`.
    assert get_tokenizer("multilingual").tokenize("única") == ["unica"]


def test_multilingual_tokenizer_collapses_regular_plurals() -> None:
    tok = get_tokenizer("multilingual")
    assert tok.tokenize("cavalos") == tok.tokenize("cavalo")
    assert tok.tokenize("homens") == tok.tokenize("homem")
    assert tok.tokenize("animais") == tok.tokenize("animal")


def test_multilingual_tokenizer_leaves_short_s_words_alone() -> None:
    # `mais` / `pais` must not be stemmed into `mai` / `pai`.
    assert "mais" in get_tokenizer("multilingual").tokenize("mais pais")


def test_unknown_tokenizer_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="tokenizer"):
        get_tokenizer("klingon")
