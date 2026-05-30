"""Cross-encoder reranker adapter (split from api/services/search.py — Task A1).

``apply_reranker`` and the template-dict adapter ``rerank_template_results``
are the two public surfaces imported by routes and tests.  Both are
re-exported on ``api.services.search`` so caller import paths are unchanged.

Monkeypatch note: tests patch ``api.services.search.search_rerank`` (the
aliased :func:`cinemateca.search.rerank` verb).  ``apply_reranker`` performs a
*lazy module-level attribute lookup* — ``import api.services.search as _svc``
inside the call body — so the patched name is always read from the live module
object and the monkeypatch is effective even though the function body lives here.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from cinemateca.search.types import Hit, Query, SearchMode, SearchResult

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _gpu_available(cfg: Any) -> bool:
    """True when the configured device resolves to CUDA or MPS.

    Used to make the reranker's ``enabled: auto`` default profile-aware:
    on by default on a GPU box, off on the CPU / HuggingFace-Spaces demo
    where the ~2.4 GB cross-encoder would make every query multi-second.
    Any probe failure (partial cfg, torch unavailable) is treated as
    CPU → reranker off, the safe/fast default.
    """
    try:
        from cinemateca.device import device_from_config

        return device_from_config(cfg).type in {"cuda", "mps"}
    except Exception:  # noqa: BLE001 — defensive: any probe failure → off
        logger.debug("reranker device probe failed; treating as CPU", exc_info=True)
        return False


def _resolve_enabled(raw: Any, cfg: Any) -> bool:
    """Coerce a ``retrieval.reranker.enabled`` value to a bool.

    Accepts a literal bool or the string ``"auto"`` (profile-aware:
    GPU-on / CPU-off via :func:`_gpu_available`). Anything else falls
    back to ``bool(raw)``.
    """
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str) and raw.strip().lower() == "auto":
        return _gpu_available(cfg)
    return bool(raw)


def _reranker_settings(cfg: Any, enabled_override: bool | None = None) -> tuple[bool, str, int]:
    """Read ``retrieval.reranker.{enabled,model,top_k_in}`` with defaults.

    Defaults: ``enabled=True``, ``model='default'``, ``top_k_in=20``. A cfg
    missing ``retrieval`` or ``retrieval.reranker`` falls back to all-defaults
    silently — callers should never see an ``AttributeError`` from a partial
    config. ``enabled`` may be a bool or the string ``"auto"`` (resolved by
    :func:`_resolve_enabled` to GPU-on / CPU-off).
    """
    rr = getattr(getattr(cfg, "retrieval", None), "reranker", None)
    if rr is None:
        enabled, model, top_k_in = True, "default", 20
    else:
        enabled = _resolve_enabled(getattr(rr, "enabled", True), cfg)
        model = str(getattr(rr, "model", "default"))
        top_k_in = int(getattr(rr, "top_k_in", 20))
    if enabled_override is not None:
        enabled = bool(enabled_override)
    return (enabled, model, top_k_in)


def reranker_default_enabled(cfg: Any) -> bool:
    """Resolve the reranker's *default* enabled state (no request override).

    This is the value the Buscar UI seeds its Rerank toggle with on a
    browser that has no saved preference: ``true`` on a GPU box,
    ``false`` on the CPU / demo profile when ``retrieval.reranker.enabled``
    is ``auto``. Pinning the config to ``true``/``false`` is honoured as-is.
    The per-browser localStorage preference (and the per-request
    ``?reranker_enabled=`` override) still win over this default.
    """
    enabled, _model, _top_k_in = _reranker_settings(cfg)
    return enabled


def apply_reranker(
    result: SearchResult, *, cfg: Any, enabled_override: bool | None = None
) -> SearchResult:
    """Apply the cross-encoder reranker to a :class:`SearchResult`.

    Reads ``retrieval.reranker.*`` from ``cfg``; ``enabled_override`` lets
    the request-level ``?reranker_enabled=`` toggle opt in/out without
    mutating global config. Safe to call unconditionally at the outermost
    boundary of any retriever path that produces a :class:`SearchResult`.
    Tests can stub the underlying verb with
    ``monkeypatch.setattr(svc, "search_rerank", ...)``.

    The verb is resolved via a lazy import of ``api.services.search`` so that
    ``monkeypatch.setattr(api.services.search, "search_rerank", ...)`` is
    visible at call time — the patched attribute is read from the live module
    object, not captured at import time.
    """
    import api.services.search as _svc  # lazy — avoids circular at import time

    enabled, model, top_k_in = _reranker_settings(cfg, enabled_override)
    if not enabled:
        return result
    return _svc.search_rerank(result, model=model, top_k_in=top_k_in)


def _result_key(row: dict[str, Any]) -> tuple[str, int]:
    return (str(row.get("film_slug") or ""), int(row.get("scene_id") or 0))


def _row_score(row: dict[str, Any]) -> float:
    raw = row.get("similarity", row.get("score", 0.0))
    try:
        return float(raw or 0.0)
    except (TypeError, ValueError):
        return 0.0


def rerank_template_results(
    results: list[dict[str, Any]],
    *,
    cfg: Any,
    query: str,
    mode: str = "hybrid",
    enabled: bool | None = None,
) -> list[dict[str, Any]]:
    """Apply the text reranker to enriched template-result dicts.

    Current HTTP dispatchers still produce DataFrames / ``list[dict]`` before
    route-level enrichment adds descriptions and tags. The cross-encoder needs
    those descriptions, so adapt the final card dicts into ``SearchResult``,
    rerank once, then return the same dict shape in reranked order.
    """
    if enabled is False or not results:
        return results

    originals: dict[tuple[str, int], dict[str, Any]] = {}
    hits: list[Hit] = []
    for row in results:
        try:
            sid = int(row.get("scene_id") or 0)
        except (TypeError, ValueError):
            continue
        key = (str(row.get("film_slug") or ""), sid)
        originals[key] = row
        tags = row.get("tags") or []
        hits.append(
            Hit(
                scene_id=sid,
                score=_row_score(row),
                keyframe_path=str(row.get("keyframe_path") or row.get("filepath") or ""),
                film_slug=key[0] or None,
                film_title=row.get("film_title"),
                timecode=str(row.get("timecode") or ""),
                description=str(row.get("description") or ""),
                tags=list(tags) if isinstance(tags, list) else [],
            )
        )
    if not hits:
        return results

    search_mode = cast(SearchMode, mode if mode in {"clip", "bm25", "hybrid"} else "hybrid")
    search_result = SearchResult(
        hits=hits,
        mode=search_mode,
        weights=None,
        query=Query.of_text(query),
        no_index=False,
    )
    try:
        reranked = apply_reranker(search_result, cfg=cfg, enabled_override=enabled)
    except Exception as exc:
        logger.warning("reranker failed; leaving text results unchanged: %s", exc)
        return results

    ordered: list[dict[str, Any]] = []
    used: set[tuple[str, int]] = set()
    for hit in reranked.hits:
        key = (hit.film_slug or "", hit.scene_id)
        original = originals.get(key)
        if original is None:
            continue
        row = dict(original)
        if hit.rerank_score is not None:
            row["rerank_score"] = hit.rerank_score
        ordered.append(row)
        used.add(key)

    if not ordered:
        return results
    # If the configured top_k_in is lower than the requested top-k, preserve
    # unscored tail results after the reranked head instead of making cards
    # disappear merely because reranking is enabled.
    ordered.extend(row for row in results if _result_key(row) not in used)
    return ordered
