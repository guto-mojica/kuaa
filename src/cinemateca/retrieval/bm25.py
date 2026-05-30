"""BM25 index per film. Pure-Python, no disk I/O.

Wraps ``rank_bm25.BM25Okapi`` with:
  * a ``build`` classmethod that accepts already-loaded
    ``descriptions`` + optional ``transcripts`` + ``tag_index`` and
    constructs the BM25 model.
  * a ``query`` method that returns ``[(scene_id, score), …]``
    instead of raw doc-index integers.

Why no disk I/O here? The merged tag-index isn't a single file on
disk — it's the result of ``api/services/catalog.py::load_tag_index``
merging ``scene_tags.json`` + the annotations file in memory.
Keeping ``BM25Index`` pure means the retrieval package has zero
dependency on the catalog service. The cache + disk + merge logic
lives in ``api/services/search.py::_get_bm25_index_for_ctx`` (Task C2).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from cinemateca.retrieval.corpus import build_corpus
from cinemateca.retrieval.tokenize import RegexTokenizer, Tokenizer

logger = logging.getLogger(__name__)


@dataclass
class BM25Index:
    """A built BM25 index for one film.

    Attributes:
        scene_ids: doc-index → scene_id mapping (parallel to ``model.doc_freqs``).
        model: the underlying ``rank_bm25.BM25Okapi`` instance, or ``None``
            when the corpus was empty (query then short-circuits to []).
        stopwords_lang: forwarded to the default ``RegexTokenizer`` at query
            time when no explicit ``tokenizer`` is stored. Kept for
            backward-compatibility with callers that use the old two-field
            constructor signature.
        tokenizer: :class:`~cinemateca.retrieval.tokenize.Tokenizer` used at
            both index-build time and query time. Stored on the dataclass so
            query reuses the exact same tokenizer that built the corpus.
    """

    scene_ids: list[int]
    model: BM25Okapi | None
    stopwords_lang: str | None
    tokenizer: Tokenizer = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        # Ensure ``tokenizer`` is always set: if the caller used the old
        # two-field positional/keyword form (scene_ids, model, stopwords_lang)
        # and omitted ``tokenizer``, synthesise the matching RegexTokenizer.
        if self.tokenizer is None:
            self.tokenizer = RegexTokenizer(stopwords_lang=self.stopwords_lang)

    @classmethod
    def build(
        cls,
        *,
        descriptions: Sequence[dict],
        tag_index: Mapping[str, Sequence[int]],
        transcripts: Sequence[dict] | None = None,
        stopwords_lang: str | None = None,
        tokenizer: Tokenizer | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> BM25Index:
        """Build a BM25 index.

        Args:
            descriptions: Scene-description dicts.
            tag_index: ``{tag: [scene_id, …]}`` mapping.
            transcripts: Optional Whisper transcript dicts.
            stopwords_lang: Legacy knob — used only when ``tokenizer`` is
                ``None`` (constructs a ``RegexTokenizer(stopwords_lang=…)``).
            tokenizer: Explicit :class:`Tokenizer` instance. When provided,
                ``stopwords_lang`` is ignored.
            k1: BM25 saturation parameter.
            b: BM25 length-normalisation parameter.
        """
        _tok: Tokenizer = (
            tokenizer if tokenizer is not None else RegexTokenizer(stopwords_lang=stopwords_lang)
        )
        docs = build_corpus(
            descriptions,
            tag_index,
            transcripts=transcripts,
            stopwords_lang=stopwords_lang,
            tokenizer=_tok,
        )
        if not docs:
            return cls(scene_ids=[], model=None, stopwords_lang=stopwords_lang, tokenizer=_tok)
        scene_ids = [sid for sid, _ in docs]
        tokenised = [tokens for _, tokens in docs]
        model = BM25Okapi(tokenised, k1=k1, b=b)
        return cls(scene_ids=scene_ids, model=model, stopwords_lang=stopwords_lang, tokenizer=_tok)

    def query(self, text: str, top_k: int) -> list[tuple[int, float]]:
        """Return top-K ``(scene_id, bm25_score)`` ranked descending.

        Empty index, non-positive top_k, or empty token query → empty list.
        Uses the same tokenizer that was used at build time.
        """
        if self.model is None or top_k <= 0:
            return []
        q_tokens = self.tokenizer.tokenize(text)
        if not q_tokens:
            return []
        scores = self.model.get_scores(q_tokens)
        # Drop zero-score docs — rank_bm25 returns them as "no match"
        # noise that would otherwise dilute the RRF fused ranking.
        ranked = sorted(
            ((self.scene_ids[i], float(score)) for i, score in enumerate(scores) if score > 0),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return ranked[:top_k]
