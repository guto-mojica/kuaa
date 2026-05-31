"""Cross-encoder reranker — bge-reranker-v2-m3 by default.

Fills in the M2 stub. ``model='noop'`` remains a passthrough escape
hatch (wider tests + UI flows that want to disable rerank without
changing config rely on it). ``model='default'`` resolves to
``BAAI/bge-reranker-v2-m3``; any other string is treated as an HF model
id and passed to the loader.

The cross-encoder is consulted only over the top ``top_k_in`` hits in
the input :class:`~cinemateca.search.types.SearchResult`; everything
beyond that rank is dropped before scoring so we never pay the
quadratic-ish cost of reranking the entire candidate set.

The query text is read from ``result.query.text``; if it is ``None``
(image-only query — text-query field is empty) the cross-encoder is
skipped and the result is returned unchanged, because a text-pair
reranker has nothing to score against an image-only query.

The config-aware wrapper (``rerank_enabled`` / ``rerank_model``) lives
one level up in :mod:`api.services.search` — see Task 3.2.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from functools import lru_cache
from typing import Protocol

from cinemateca.models.manifest import ModelCard, get_card
from cinemateca.search.types import Hit, SearchResult

logger = logging.getLogger(__name__)

#: Provenance for the cross-encoder reranker backend (manifest single source
#: of truth, C10/F6). The reranker has no ``models.*`` config selector (it is
#: configured under ``retrieval.reranker.*``), so it is a module-level link
#: rather than a class attribute; ``registry.model_card(settings, "reranker")``
#: resolves to this same card.
CARD: ModelCard = get_card("bge_reranker_v2_m3")

_DEFAULT_MODEL_ID = "BAAI/bge-reranker-v2-m3"
DEFAULT_TOP_K_IN = 20


class _RerankerLike(Protocol):
    """Minimal protocol satisfied by FlagEmbedding's ``FlagReranker``."""

    def compute_score(self, pairs: list[list[str]]) -> list[float]: ...


@lru_cache(maxsize=2)
def _load_reranker(model_id: str) -> _RerankerLike:
    """Lazy + cached cross-encoder load.

    Raises ``RuntimeError`` (not ``ImportError``) when the optional
    ``FlagEmbedding`` extra is missing so callers see a single, actionable
    failure mode regardless of which dep happens to be absent.

    ``use_fp16`` follows CUDA availability — fp16 is materially faster on
    GPU and supported by bge-reranker-v2-m3; on CPU/MPS the cost is the
    same or worse (no half-precision matmul kernels) so we stay fp32.
    """
    try:
        from FlagEmbedding import FlagReranker
    except ImportError as exc:
        raise RuntimeError(
            "Cross-encoder reranker requires the FlagEmbedding package. "
            "Install with: uv pip install FlagEmbedding"
        ) from exc
    use_fp16 = False
    try:
        import torch

        use_fp16 = bool(torch.cuda.is_available())
    except ImportError:
        pass
    logger.info("Loading cross-encoder reranker: %s (fp16=%s)", model_id, use_fp16)
    return FlagReranker(model_id, use_fp16=use_fp16)


def rerank(
    result: SearchResult,
    *,
    model: str = "default",
    top_k_in: int = DEFAULT_TOP_K_IN,
) -> SearchResult:
    """Cross-encoder rerank the top ``top_k_in`` hits in ``result``.

    Args:
        result: input :class:`SearchResult`. The query text is read from
            ``result.query.text``; image-only queries (``text is None``)
            short-circuit to a passthrough since the text-pair reranker
            has nothing to score.
        model: ``"noop"`` is a documented passthrough escape hatch used by
            tests and any UI flow that wants to disable rerank without
            mutating config. ``"default"`` resolves to
            ``BAAI/bge-reranker-v2-m3``. Any other string is treated as an
            HF model id and forwarded to the loader.
        top_k_in: max number of hits to send to the cross-encoder. Hits
            beyond this rank are dropped from the returned result.

    Returns:
        A new :class:`SearchResult` with at most ``top_k_in`` hits, each
        carrying a populated ``rerank_score`` float, sorted by descending
        ``rerank_score``. All other ``SearchResult`` fields (``mode``,
        ``weights``, ``query``, ``no_index``) are preserved.
    """
    if model == "noop":
        return result
    if not result.hits:
        return result

    query_text = result.query.text
    if not query_text:
        # Image-only (or empty) query — nothing to feed the cross-encoder.
        logger.debug("rerank skipped: query has no text component")
        return result

    model_id = _DEFAULT_MODEL_ID if model == "default" else model
    reranker = _load_reranker(model_id)
    top = result.hits[:top_k_in]
    pairs = [[query_text, (h.description or "")] for h in top]

    scores = reranker.compute_score(pairs)
    rescored: list[Hit] = [
        replace(h, rerank_score=float(s)) for h, s in zip(top, scores, strict=True)
    ]
    rescored.sort(key=lambda h: h.rerank_score or 0.0, reverse=True)
    return replace(result, hits=rescored, reranker_applied=True)
