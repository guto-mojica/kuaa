"""BM25 index per film. Pure-Python, no disk I/O.

Wraps ``rank_bm25.BM25Okapi`` with:
  * a ``build`` classmethod that accepts already-loaded
    ``descriptions`` + ``tag_index`` and constructs the BM25 model.
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
from collections.abc import Sequence
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from cinemateca.retrieval.corpus import build_corpus

logger = logging.getLogger(__name__)


@dataclass
class BM25Index:
    """A built BM25 index for one film.

    Attributes:
        scene_ids: doc-index → scene_id mapping (parallel to ``model.doc_freqs``).
        model: the underlying ``rank_bm25.BM25Okapi`` instance, or ``None``
            when the corpus was empty (query then short-circuits to []).
        stopwords_lang: forwarded to the tokenizer at query time.
    """

    scene_ids: list[int]
    model: BM25Okapi | None
    stopwords_lang: str | None

    @classmethod
    def build(
        cls,
        *,
        descriptions: Sequence[dict],
        tag_index: dict[str, Sequence[int]],
        stopwords_lang: str | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> BM25Index:
        docs = build_corpus(descriptions, tag_index, stopwords_lang=stopwords_lang)
        if not docs:
            return cls(scene_ids=[], model=None, stopwords_lang=stopwords_lang)
        scene_ids = [sid for sid, _ in docs]
        tokenised = [tokens for _, tokens in docs]
        model = BM25Okapi(tokenised, k1=k1, b=b)
        return cls(scene_ids=scene_ids, model=model, stopwords_lang=stopwords_lang)

    def query(self, text: str, top_k: int) -> list[tuple[int, float]]:
        """Return top-K ``(scene_id, bm25_score)`` ranked descending.

        Empty index, non-positive top_k, or empty token query → empty list.
        """
        if self.model is None or top_k <= 0:
            return []
        from cinemateca.retrieval.tokenize import tokenize

        q_tokens = tokenize(text, stopwords_lang=self.stopwords_lang)
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
