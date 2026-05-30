"""Pure tokenizer used by BM25 corpus build + query path.

The default config ships without stopword removal — PT stopwords overlap
with cinematographically meaningful words (`sem`, `não`) and lossy-
defaults risk more than they save. Callers may opt in via
``stopwords_lang="pt"`` (requires ``nltk`` installed; not a hard dep).

Pluggable tokenizer classes
----------------------------
``Tokenizer`` (Protocol, runtime-checkable) — single ``tokenize(text) -> list[str]``.
``RegexTokenizer`` — wraps the module-level ``tokenize`` function (default).
``MultilingualTokenizer`` — PT-aware: enables PT stopword removal.
``get_tokenizer(name)`` — resolve by config name (``regex`` | ``multilingual``).
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Protocol, runtime_checkable

# Unicode letter or digit, run-length 1+. \w in Python 3 with the
# default re flags is Unicode-aware and covers PT diacritics
# (ç, ã, é, …) without pulling in the external `regex` package.
# We anchor to letters + digits explicitly so underscores and emoji
# punctuation do not slip through as tokens.
_TOKEN_RE = re.compile(r"[^\W\d_]+|\d+", flags=re.UNICODE)


@lru_cache(maxsize=1)
def _pt_stopwords() -> frozenset[str]:
    """Lazy-load PT stopwords from nltk.

    Cached so repeated calls don't re-import. If nltk isn't installed,
    raise ImportError eagerly — callers can guard with importorskip in
    tests or check the config flag in production.
    """
    from nltk.corpus import stopwords  # type: ignore[import-not-found]

    return frozenset(stopwords.words("portuguese"))


def tokenize(text: str, *, stopwords_lang: str | None = None) -> list[str]:
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

    Args:
        text: Input string. Empty/whitespace → empty list, no crash.
        stopwords_lang: ISO 639-1 language code (currently only ``"pt"``
            is recognised). ``None`` (default) disables stopword removal.

    Returns:
        List of token strings, in document order.
    """
    if not text:
        return []
    normalised = unicodedata.normalize("NFKC", text).lower()
    tokens = [t for t in _TOKEN_RE.findall(normalised) if _keep_token(t)]
    if stopwords_lang == "pt":
        sw = _pt_stopwords()
        tokens = [t for t in tokens if t not in sw]
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

    def __init__(self, *, stopwords_lang: str | None = None) -> None:
        self._stopwords_lang = stopwords_lang

    def tokenize(self, text: str) -> list[str]:
        """Tokenize ``text`` using the configured stopwords language (if any)."""
        return tokenize(text, stopwords_lang=self._stopwords_lang)


class MultilingualTokenizer(RegexTokenizer):
    """PT-aware tokenizer matching the SigLIP-multilingual visual story.

    Adds PT-stopword removal by default so PT queries don't dilute the
    BM25 ranking with function words.
    """

    def __init__(self) -> None:
        super().__init__(stopwords_lang="pt")


def get_tokenizer(name: str) -> Tokenizer:
    """Resolve a tokenizer by config name (``regex`` | ``multilingual``).

    Args:
        name: One of ``"regex"`` (default, no stopword removal) or
            ``"multilingual"`` (PT-aware stopword removal).

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
