"""Pure tokenizer used by BM25 corpus build + query path.

The bare ``tokenize`` default is deliberately conservative: no stopword
removal, accents preserved, no suffix stripping. Every normalisation step
is opt-in, because each one is lossy and the archive corpus is small
enough that recall matters more than index size.

Portuguese notes
----------------
Stopword removal uses a *reduced* PT list (:data:`_PT_STOPWORDS`) rather
than the full nltk set. The full set removes ``sem`` and ``não``, which
are cinematographically meaningful here — the pipeline itself writes the
tag ``sem-pessoas``, so dropping ``sem`` would make the system unable to
match a tag it generated.

Diacritic folding and suffix stripping must be applied *symmetrically* at
index-build and query time or they cause misses rather than fix them.
That is why they are properties of the ``Tokenizer`` instance (which is
stored on the index and reused at query time) rather than call-site flags.

Pluggable tokenizer classes
----------------------------
``Tokenizer`` (Protocol, runtime-checkable) — single ``tokenize(text) -> list[str]``.
``RegexTokenizer`` — wraps the module-level ``tokenize`` function (default).
``MultilingualTokenizer`` — PT-aware: reduced stopwords + diacritic folding
    + light suffix normalisation.
``get_tokenizer(name)`` — resolve by config name (``regex`` | ``multilingual``).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Protocol, runtime_checkable

# Unicode letter or digit, run-length 1+. \w in Python 3 with the
# default re flags is Unicode-aware and covers PT diacritics
# (ç, ã, é, …) without pulling in the external `regex` package.
# We anchor to letters + digits explicitly so underscores and emoji
# punctuation do not slip through as tokens.
_TOKEN_RE = re.compile(r"[^\W\d_]+|\d+", flags=re.UNICODE)


# Reduced Portuguese stopword list — articles, prepositions, conjunctions,
# copulas and pronouns only. Deliberately EXCLUDES negation and absence
# words (``não``, ``nem``, ``sem``, ``nunca``) and quantity words (``mais``,
# ``menos``, ``muito``, ``pouco``), which carry meaning when describing a
# shot. ``sem`` in particular is load-bearing: the describer emits the tag
# ``sem-pessoas`` for empty frames.
#
# Inlined rather than pulled from nltk: nltk is not a project dependency,
# and the lazy ``from nltk.corpus import stopwords`` this replaces raised
# ModuleNotFoundError at corpus-build time — which the caller in
# ``kuaa.search.aggregate`` does not catch, turning the config knob
# ``bm25.stopwords_lang: pt`` into a 500.
_PT_STOPWORDS: frozenset[str] = frozenset(
    """
    as os um uma uns umas ao aos da das do dos na nas no nos
    de em por para com sobre entre até desde após ante
    que se como quando onde qual quais
    eu tu ele ela nos vos eles elas me te lhe lhes
    meu minha teu tua seu sua nosso nossa
    este esta esse essa aquele aquela isto isso aquilo
    ser sao foi era eram sendo estar esta estao estava
    ter tem tinha haver ha havia
    e ou mas porem porque pois entao tambem ja
    """.split()
)

# Portuguese plural → singular rewrites, longest-suffix-first. Only the
# regular patterns are here; irregular plurals are carried as explicit
# surface forms in the bilingual lexicon instead of being stemmed.
_PT_PLURAL_RULES: tuple[tuple[str, str], ...] = (
    ("oes", "ao"),  # coracoes → coracao
    ("aes", "ao"),  # caes → cao
    ("aos", "ao"),  # maos → mao
    ("ais", "al"),  # animais → animal
    ("eis", "el"),  # papeis → papel
    ("ois", "ol"),  # lencois → lencol
    ("uis", "ul"),  # pauis → paul
    ("ns", "m"),  # homens → homem, jovens → jovem
)


def fold_diacritics(token: str) -> str:
    """Strip combining marks: ``única`` → ``unica``, ``cão`` → ``cao``.

    NFKC (used by :func:`tokenize`) is compatibility *composition* and
    leaves accents intact, so it does not bridge the gap between the
    unaccented tags the pipeline writes (``pessoa-unica``) and the
    accented spelling a Portuguese curator types (``única``).
    """
    decomposed = unicodedata.normalize("NFD", token)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalise_suffix(token: str) -> str:
    """Collapse regular Portuguese/English plurals to a shared stem.

    Expects an already diacritic-folded token. Conservative by design:
    tokens of 4 characters or fewer are returned untouched, so short
    words that merely end in ``s`` (``mais``, ``pais``, ``was``) survive.

    This is a normaliser, not a linguist's stemmer — it only has to be
    *consistent* between index and query to earn its recall. Irregular
    forms it gets wrong (``mulheres`` → ``mulhere``) are handled by
    listing both surface forms in the bilingual lexicon.
    """
    if len(token) <= 4:
        return token
    for suffix, replacement in _PT_PLURAL_RULES:
        if token.endswith(suffix):
            return token[: -len(suffix)] + replacement
    if token.endswith("s"):
        return token[:-1]
    return token


def tokenize(
    text: str,
    *,
    stopwords_lang: str | None = None,
    fold: bool = False,
    stem: bool = False,
) -> list[str]:
    """Tokenise text for BM25 indexing/querying.

    Steps:
      1. NFKC-normalise (collapses fullwidth / ligatures).
      2. Lowercase.
      3. Extract Unicode-letter runs + integer runs.
      4. Drop length-1 *ASCII* tokens (`a`, `e`, `o`, `i`, …).
         Length-1 *non-ASCII* letters (`é`, `à`, `ó`) are kept because
         they are rare-but-semantic in PT (e.g., `é` = the verb "is")
         and adding them costs near-nothing in index size.
      5. Optionally drop tokens in the configured language's stopword list.
      6. Optionally fold diacritics, then optionally normalise plurals.

    Args:
        text: Input string. Empty/whitespace → empty list, no crash.
        stopwords_lang: ISO 639-1 language code (currently only ``"pt"``
            is recognised). ``None`` (default) disables stopword removal.
        fold: Strip combining marks (see :func:`fold_diacritics`).
        stem: Collapse regular plurals (see :func:`normalise_suffix`).
            Applied after folding, which it assumes.

    Returns:
        List of token strings, in document order. Defaults are lossless
        relative to the original tokenizer — every normalisation is opt-in.
    """
    if not text:
        return []
    normalised = unicodedata.normalize("NFKC", text).lower()
    tokens = [t for t in _TOKEN_RE.findall(normalised) if _keep_token(t)]
    if stopwords_lang == "pt":
        # Match stopwords on the folded form so `não`/`nao` behave alike.
        tokens = [t for t in tokens if fold_diacritics(t) not in _PT_STOPWORDS]
    if fold:
        tokens = [fold_diacritics(t) for t in tokens]
    if stem:
        tokens = [normalise_suffix(t) for t in tokens]
    return tokens


def _keep_token(t: str) -> bool:
    """Drop length-1 ASCII tokens; keep everything else.

    See ``tokenize`` docstring for rationale.
    """
    if len(t) >= 2:
        return True
    return not t.isascii()


# ── Pluggable tokenizer Protocol + implementations ───────────────────────────


@runtime_checkable
class Tokenizer(Protocol):
    """Tokenises text for BM25 indexing/querying."""

    def tokenize(self, text: str) -> list[str]: ...


class RegexTokenizer:
    """Default tokenizer — wraps the legacy module-level ``tokenize``.

    Unicode NFKC + lowercase + letter/digit runs; length-1 ASCII dropped.
    """

    def __init__(
        self,
        *,
        stopwords_lang: str | None = None,
        fold: bool = False,
        stem: bool = False,
    ) -> None:
        self._stopwords_lang = stopwords_lang
        self._fold = fold
        self._stem = stem

    def tokenize(self, text: str) -> list[str]:
        """Tokenize ``text`` using this instance's configured normalisation."""
        return tokenize(
            text,
            stopwords_lang=self._stopwords_lang,
            fold=self._fold,
            stem=self._stem,
        )


class MultilingualTokenizer(RegexTokenizer):
    """PT-aware tokenizer matching the SigLIP-multilingual visual story.

    Enables the full normalisation chain: reduced PT stopwords, diacritic
    folding, and regular-plural collapsing. Because the same instance is
    stored on :class:`~kuaa.retrieval.bm25.BM25Index` and reused at query
    time, index and query are normalised identically by construction.
    """

    def __init__(self) -> None:
        super().__init__(stopwords_lang="pt", fold=True, stem=True)


def get_tokenizer(name: str) -> Tokenizer:
    """Resolve a tokenizer by config name (``regex`` | ``multilingual``).

    Args:
        name: One of ``"regex"`` (conservative, no normalisation) or
            ``"multilingual"`` (PT-aware stopwords + folding + stemming).

    Returns:
        A :class:`Tokenizer`-protocol-conforming instance.

    Raises:
        ValueError: If ``name`` is not a recognised tokenizer id.
    """
    if name == "regex":
        return RegexTokenizer()
    if name == "multilingual":
        return MultilingualTokenizer()
    raise ValueError(f"Unknown bm25 tokenizer: {name!r}")
