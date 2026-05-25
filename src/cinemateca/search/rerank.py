"""Cross-encoder reranker — M2 implementation.

Stage-2 of the retrieval pipeline: takes the top-K hits from the first-stage
(CLIP / BM25 / hybrid) and re-scores each (query, scene_description) pair with
a bi-directional cross-encoder. Unlike bi-encoders (CLIP, BM25) the cross-
encoder sees both query and document simultaneously, enabling finer relevance
judgements at the cost of per-query inference — which is why it only runs on
the pre-filtered top-K candidates rather than the full index.

Default model: cross-encoder/ms-marco-MiniLM-L-12-v2 (~70 MB).
  * Initialised once per process and cached in ``_CE_CACHE``.
  * ``model="noop"`` is a passthrough — preserves the P1 test-escape hatch.

Two public entry points:
  * ``rerank(result, *, film, top_k, model)`` — typed ``SearchResult`` → ``SearchResult``.
  * ``rerank_dataframe(df, *, query, metadata_dir, top_k, model)`` — DataFrame → DataFrame,
    used by ``api/services/search.py`` which works in the DataFrame layer.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from cinemateca.search.types import SearchResult

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"

# Module-level model cache: model_name → CrossEncoder instance.
_CE_CACHE: dict[str, Any] = {}


def _get_cross_encoder(model_name: str) -> Any:
    if model_name not in _CE_CACHE:
        from sentence_transformers import CrossEncoder  # lazy import — model not loaded at startup

        logger.info("rerank: loading cross-encoder %r (first call only)", model_name)
        _CE_CACHE[model_name] = CrossEncoder(model_name)
    return _CE_CACHE[model_name]


def _load_descriptions(metadata_dir: Path) -> dict[int, str]:
    """Return {scene_id: description} from scene_descriptions.json."""
    path = metadata_dir / "scene_descriptions.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        logger.warning("rerank: could not read %s", path)
        return {}
    out: dict[int, str] = {}
    for entry in data if isinstance(data, list) else []:
        sid = entry.get("scene_id")
        if sid is None:
            continue
        try:
            out[int(sid)] = str(entry.get("description") or "")
        except (TypeError, ValueError):
            continue
    return out


def rerank(
    result: SearchResult,
    *,
    film: Any,
    top_k: int = 10,
    model: str = "default",
) -> SearchResult:
    """Rerank a ``SearchResult`` with a cross-encoder.

    ``model="noop"`` returns the input unchanged (P1 test-escape hatch).
    Image queries (``result.query.text is None``) are returned unchanged —
    the cross-encoder is text-only.
    """
    if model == "noop":
        return result
    if result.query.text is None:
        return result
    if not result.hits:
        return result

    model_name = _DEFAULT_MODEL if model == "default" else model
    descriptions = _load_descriptions(film.metadata_dir)
    query_text = result.query.text

    pairs = [
        [query_text, descriptions.get(hit.scene_id, hit.description or "")]
        for hit in result.hits
    ]

    try:
        scores: list[float] = _get_cross_encoder(model_name).predict(pairs).tolist()
    except Exception:
        logger.exception("rerank: cross-encoder failed; returning original order")
        return result

    reranked = [hit for hit, _ in sorted(zip(result.hits, scores), key=lambda x: x[1], reverse=True)]
    return SearchResult(
        hits=reranked[:top_k],
        mode=result.mode,
        weights=result.weights,
        query=result.query,
        no_index=result.no_index,
    )


def rerank_dataframe(
    df: Any,  # pd.DataFrame — typed as Any to avoid a hard pandas import at the module level
    *,
    query: str,
    metadata_dir: Path,
    top_k: int = 10,
    model: str = "default",
) -> Any:
    """Rerank a result DataFrame by cross-encoder score.

    Convenience wrapper for ``api/services/search.py``, which works in
    the DataFrame layer rather than the typed ``SearchResult`` layer.
    Returns the input unchanged when: ``model="noop"``, empty DataFrame,
    empty query, or ``scene_id`` column absent.
    """
    if model == "noop" or not query:
        return df
    try:
        if df.empty or "scene_id" not in df.columns:
            return df
    except AttributeError:
        return df

    model_name = _DEFAULT_MODEL if model == "default" else model
    descriptions = _load_descriptions(metadata_dir)

    pairs = [
        [query, descriptions.get(int(row["scene_id"]), "")]
        for _, row in df.iterrows()
    ]

    try:
        scores = _get_cross_encoder(model_name).predict(pairs)
    except Exception:
        logger.exception("rerank_dataframe: cross-encoder failed; returning original order")
        return df.head(top_k).reset_index(drop=True)

    df = df.copy()
    df["_rerank_score"] = scores
    df = df.sort_values("_rerank_score", ascending=False).drop(columns=["_rerank_score"])
    return df.head(top_k).reset_index(drop=True)
