"""Audio + fusion HTTP dispatchers (split from api/services/search.py — Task A1).

Thin adapters over the typed core verbs in ``cinemateca.search.audio`` /
``cinemateca.search.fusion``. The core verbs (``find_audio`` /
``aggregate_audio`` / ``find_fusion`` / ``aggregate_fusion``) own the search +
per-query metadata (C9) and return a typed :class:`SearchResult` for
programmatic consumers (eval). These HTTP dispatchers return the ``list[dict]``
view rows the ``.b-card`` template path expects, so the HTTP response shape is
unchanged.

* Audio: delegates fully to the core verb, then projects ``SearchResult.hits``
  to view rows (audio rows need only ``scene_id`` / ``score`` / ``film_slug`` /
  ``film_title``).
* Fusion: the view rows additionally carry the per-modality diagnostic
  decomposition (``clip_score`` / ``clap_score``) which is *not* part of the
  core ``Hit`` (modality-specific — kept off the shared type). The fusion
  dispatcher therefore builds its rich rows via the same core loader the verb
  uses (``cinemateca.search.fusion._fusion_rows_for_film``) rather than lifting
  from ``Hit``. The typed metadata surface for fusion is the core verb, tested
  directly.

Re-exported on ``api.services.search`` so route import paths are unchanged.
``audio_hits_to_template_dicts`` lives in ``_search_hits`` (G1 fix) and is
re-exported here so any direct ``_search_dispatch`` import sites keep working.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Re-export for backward compat — function moved to _search_hits (G1 LOC fix).
from api.services._search_hits import (  # noqa: F401
    audio_hits_to_template_dicts as audio_hits_to_template_dicts,
)


def _audio_embedder_factory(cfg: Any) -> Any:
    """``cfg -> AudioEmbedder`` (lazy registry import keeps the test seam at
    ``cinemateca.models.registry.get_audio_embedder``)."""
    from cinemateca.models.registry import get_audio_embedder

    return get_audio_embedder(cfg, device=None)


def _image_embedder_factory(cfg: Any) -> Any:
    """``cfg -> ImageEmbedder`` (lazy registry import, same seam rationale)."""
    from cinemateca.models.registry import get_image_embedder

    return get_image_embedder(cfg, device=None)


def _result_to_view_rows(result: Any) -> list[dict]:
    """Project a typed :class:`SearchResult` to the ``{scene_id, score,
    film_slug, film_title}`` view rows the template path consumes.

    Display-only / per-modality fields are not on the core ``Hit``; the fusion
    dispatcher overlays its sub-scores separately (see ``_fusion_view_rows``).
    """
    rows: list[dict] = []
    for h in result.hits:
        row: dict[str, Any] = {"scene_id": h.scene_id, "score": h.score}
        if h.film_slug is not None:
            row["film_slug"] = h.film_slug
        if h.film_title is not None:
            row["film_title"] = h.film_title
        rows.append(row)
    return rows


def dispatch_audio_search(
    cfg: Any,
    ctx: Any | None,
    q: str,
    top_k: int,
) -> tuple[list[dict], bool]:
    """Run CLAP search; return ``(hits, no_index)``.

    ``ctx`` given → per-film (:func:`cinemateca.search.audio.find_audio`);
    ``ctx=None`` → cross-film aggregate
    (:func:`cinemateca.search.audio.aggregate_audio`). ``no_index=True`` when no
    CLAP index exists. Embedder loaded at most once. Test seam: monkeypatch
    ``cinemateca.models.registry.get_audio_embedder``.
    """
    from cinemateca.search.audio import aggregate_audio, find_audio, load_audio_index

    if ctx is not None:
        audio_dir = Path(ctx.metadata_dir).parent / "audio"
        index = load_audio_index(audio_dir)
        if index is None:
            return [], True
        result = find_audio(
            index,
            _audio_embedder_factory(cfg),
            q,
            film_slug=ctx.slug,
            top_k=top_k,
        )
        return _result_to_view_rows(result), result.no_index

    result = aggregate_audio(cfg, _audio_embedder_factory, q, top_k=top_k)
    if result.no_index:
        return [], True
    return _result_to_view_rows(result), False


def dispatch_fusion_search(
    cfg: Any,
    ctx: Any | None,
    q: str,
    top_k: int,
    *,
    visual_weight: float = 0.5,
    k_each: int = 50,
) -> tuple[list[dict], bool]:
    """CLIP×CLAP fusion search; returns ``(hits, no_index)``.

    ``ctx`` given → per-film; ``ctx=None`` → cross-film aggregate over the
    registry. Missing modalities contribute zero rows. View rows are the rich
    ``search_fusion`` rows (carry the per-modality ``clip_score`` /
    ``clap_score`` decomposition that is intentionally *not* on the core
    ``Hit``), tagged with ``film_slug`` / ``film_title``. Built via the core
    loader :func:`cinemateca.search.fusion._fusion_rows_for_film` so the api
    layer keeps no duplicate of the index-loading logic; the typed
    metadata-bearing :class:`SearchResult` surface is the core verb. Aggregate
    derives paths directly from ``cfg.paths.library_dir`` to avoid
    ``FilmContext.for_film`` side-effects (mkdir); embedders load at most once.
    Test seam: monkeypatch the registry embedder factories.
    """
    from cinemateca.library import scan_library
    from cinemateca.search.fusion import _fusion_rows_for_film

    if ctx is not None:
        rows, no_index, _, _ = _fusion_rows_for_film(
            cfg=cfg,
            embeddings_dir=ctx.embeddings_dir,
            audio_dir=Path(ctx.metadata_dir).parent / "audio",
            query_text=q,
            top_k=top_k,
            visual_weight=visual_weight,
            k_each=k_each,
            clip_embedder=None,
            clap_embedder=None,
            image_embedder_factory=_image_embedder_factory,
            audio_embedder_factory=_audio_embedder_factory,
        )
        if no_index:
            return [], True
        for r in rows:
            r["film_slug"] = ctx.slug
        return rows, False

    library_dir = Path(cfg.paths.library_dir)
    films = list(scan_library(library_dir))
    if not films:
        return [], True
    clip_embedder: Any | None = None
    clap_embedder: Any | None = None
    all_hits: list[dict] = []
    any_film = False
    for film in films:
        film_rows, film_no_index, clip_embedder, clap_embedder = _fusion_rows_for_film(
            cfg=cfg,
            embeddings_dir=library_dir / film.slug / "embeddings",
            audio_dir=library_dir / film.slug / "audio",
            query_text=q,
            top_k=top_k,
            visual_weight=visual_weight,
            k_each=k_each,
            clip_embedder=clip_embedder,
            clap_embedder=clap_embedder,
            image_embedder_factory=_image_embedder_factory,
            audio_embedder_factory=_audio_embedder_factory,
        )
        if film_no_index:
            continue
        any_film = True
        for r in film_rows:
            r["film_slug"] = film.slug
            r["film_title"] = film.title
            all_hits.append(r)
    if not any_film:
        return [], True
    all_hits.sort(key=lambda r: r["score"], reverse=True)
    return all_hits[:top_k], False
