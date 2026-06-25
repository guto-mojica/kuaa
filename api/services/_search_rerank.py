"""Cross-encoder reranker boundary (split from api/services/search.py — Task A1).

``apply_reranker`` is the typed verb wrapper: it reads
``retrieval.reranker.*`` off ``cfg`` and forwards a :class:`SearchResult`
to :func:`kuaa.search.rerank`. It is re-exported on
``api.services.search`` so caller import paths are unchanged.

C5: the production text-search path now carries a typed :class:`SearchResult`
from enrichment through rerank to the render boundary, so the old
``dict → SearchResult → dict`` adapter (``rerank_template_results``) is gone.
The two thin boundary helpers that replace it —
:func:`cards_to_result` (lift enriched card dicts to a typed result) and
:func:`result_to_cards` (project the reranked result back to card dicts) —
make the typed result the through-line rather than something rerank
reconstructs and discards on every call.

Monkeypatch note: tests patch ``api.services.search.search_rerank`` (the
aliased :func:`kuaa.search.rerank` verb).  ``apply_reranker`` performs a
*lazy module-level attribute lookup* — ``import api.services.search as _svc``
inside the call body — so the patched name is always read from the live module
object and the monkeypatch is effective even though the function body lives here.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from kuaa.search.types import Hit, Query, SearchMode, SearchResult

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
        from kuaa.device import device_from_config

        return device_from_config(cfg).type in {"cuda", "mps"}
    except Exception:  # noqa: BLE001 — defensive: any probe failure → off
        logger.debug("reranker device probe failed; treating as CPU", exc_info=True)
        return False


def _resolve_enabled(raw: Any, cfg: Any) -> bool:
    """Coerce a ``retrieval.reranker.enabled`` value to a bool.

    Accepts a literal bool or the string ``"auto"`` (profile-aware:
    GPU-on / CPU-off via :func:`_gpu_available`). Anything else falls
    back to ``bool(raw)``.

    The GPU probe is read via a lazy lookup of
    ``api.services.search._gpu_available`` — the same facade-patch convention
    :func:`apply_reranker` uses for ``search_rerank`` — so tests that
    ``monkeypatch.setattr(api.services.search, "_gpu_available", ...)`` are honoured.
    """
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str) and raw.strip().lower() == "auto":
        import api.services.search as _svc  # lazy — patchable facade lookup

        return _svc._gpu_available(cfg)
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


def _card_key(row: dict[str, Any]) -> tuple[str, int]:
    return (str(row.get("film_slug") or ""), int(row.get("scene_id") or 0))


def _card_score(row: dict[str, Any]) -> float:
    raw = row.get("similarity", row.get("score", 0.0))
    try:
        return float(raw or 0.0)
    except (TypeError, ValueError):
        return 0.0


def cards_to_result(
    cards: list[dict[str, Any]],
    *,
    query: str,
    mode: str = "hybrid",
) -> tuple[SearchResult, dict[tuple[str, int], dict[str, Any]]]:
    """Lift enriched template-card dicts into a typed :class:`SearchResult`.

    The *single* dict→typed boundary on the text-search path (C5). Enrichment
    reads descriptions/tags as JSON dicts, so the card list is where typed
    :class:`Hit` objects are built once. The returned ``originals`` map (keyed
    by ``(film_slug, scene_id)``) lets :func:`result_to_cards` re-emit the
    exact template dicts in result order — every display-only field the
    ``.b-card`` template reads (``img_url`` / ``similarity`` / ``pin_count``)
    lives there, not on the core ``Hit``. The ``SearchResult`` is the
    through-line: the caller passes it to :func:`rerank_search_result` then
    :func:`result_to_cards`, so the reranker operates on the result the path
    built rather than reconstructing one per call.
    """
    originals: dict[tuple[str, int], dict[str, Any]] = {}
    hits: list[Hit] = []
    for row in cards:
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
                score=_card_score(row),
                keyframe_path=str(row.get("keyframe_path") or row.get("filepath") or ""),
                film_slug=key[0] or None,
                film_title=row.get("film_title"),
                timecode=str(row.get("timecode") or ""),
                description=str(row.get("description") or ""),
                tags=list(tags) if isinstance(tags, list) else [],
            )
        )
    search_mode = cast(SearchMode, mode if mode in {"clip", "bm25", "hybrid"} else "hybrid")
    result = SearchResult(
        hits=hits,
        mode=search_mode,
        weights=None,
        query=Query.of_text(query),
        no_index=False,
    )
    return result, originals


def result_to_cards(
    result: SearchResult,
    originals: dict[tuple[str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Project a (possibly reranked) :class:`SearchResult` back to card dicts.

    Re-emits the original enriched template dict for each ``Hit`` in result
    order, attaching ``rerank_score`` when the cross-encoder set it. Cards the
    result dropped (e.g. ranks beyond ``top_k_in`` once reranking is enabled)
    are appended after the head in their original order, so enabling rerank
    never makes cards disappear merely for being below the input window.
    """
    ordered: list[dict[str, Any]] = []
    used: set[tuple[str, int]] = set()
    for hit in result.hits:
        key = (hit.film_slug or "", hit.scene_id)
        original = originals.get(key)
        if original is None:
            continue
        row = dict(original)
        if hit.rerank_score is not None:
            row["rerank_score"] = hit.rerank_score
        ordered.append(row)
        used.add(key)
    # Preserve the original card order for the unscored tail (originals is
    # insertion-ordered to match the input ``cards`` list).
    ordered.extend(dict(row) for key, row in originals.items() if key not in used)
    return ordered


def rerank_search_result(
    result: SearchResult,
    *,
    cfg: Any,
    enabled: bool | None = None,
) -> SearchResult:
    """Rerank a typed text :class:`SearchResult` at the render boundary.

    Thin wrapper over :func:`apply_reranker` owning the two policy bits the
    render layer should not repeat: a no-op short-circuit for an empty result
    or ``enabled=False``, and a defensive ``try/except`` so a cross-encoder
    load/scoring failure degrades to the first-stage ranking instead of
    500-ing the request. The typed result flows straight through.
    """
    if enabled is False or not result.hits:
        return result
    try:
        return apply_reranker(result, cfg=cfg, enabled_override=enabled)
    except Exception as exc:  # noqa: BLE001 — degrade to first-stage ranking
        logger.warning("reranker failed; leaving text results unchanged: %s", exc)
        return result
