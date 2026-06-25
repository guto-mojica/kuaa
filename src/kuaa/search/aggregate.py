"""Cross-film aggregate search — global retrieval lists over the library.

Relocated from ``api/services/search.py`` during P1 / T11. The function
walks every registered film, builds per-film CLIP + (optional) BM25
ranked lists, concatenates them into two GLOBAL ranked lists keyed by
``(film_slug, scene_id)``, then dispatches by retriever mode:

  * ``"clip"``   — global CLIP list, sorted by cosine.
  * ``"bm25"``   — global BM25 list, sorted by raw BM25 score.
  * ``"hybrid"`` — weighted RRF fusion of the two global lists.

Signature note: the function preserves the EXACT ``cfg``-taking, keyword-
only signature that lived in ``api.services.search`` so the existing
route call site (``api/routes/search.py``) and the 18 cross-film tests
(``test_multi_film_search.py``, ``test_aggregate_search_hybrid.py``,
``test_p1_search_snapshot.py``) keep passing byte-identical. A public
typed ``aggregate(query, *, cfg, ...)`` wrapper lands in T13 — verbatim
move first, signature reshape behind a stable public surface second.

T13 (P3.D.1): ``_get_embedder``, ``_get_search_index``, and
``has_indexed_films`` now live here. ``api.services.search`` re-exports
them for backward compatibility. Tests should monkeypatch
``kuaa.search.aggregate._get_embedder`` / ``_get_search_index``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kuaa.config import Settings
from kuaa.library import (
    FilmContext,
    derive_fps,
    load_json,
    load_tag_index,
)
from kuaa.retrieval.hybrid import DEFAULT_RRF_K
from kuaa.scene_ids import normalize_tag_index, scene_id_key
from kuaa.search._aggregate.film_filter import CandidateFilm, FilmFilter
from kuaa.search._aggregate.fusion import fuse_global_rrf as _fuse_rrf_many
from kuaa.search._aggregate.materialize import materialize_hits
from kuaa.search._aggregate.scorers import BM25Scorer, CLIPScorer, MetadataScorer
from kuaa.search.cache import IndexStatus, SearchIndex, load_index
from kuaa.search.types import (
    Filters,
    Hit,
    HybridWeights,
    Query,
    SearchMode,
    SearchResult,
)

logger = logging.getLogger(__name__)

# Canonical filenames for the per-film CLIP index. Mirror
# ``config/default.yaml`` → ``embeddings.*``; used as defaults when
# ``cfg.embeddings`` is absent (unit tests with minimal configs).
_DEFAULT_EMBEDDINGS_FILENAME = "keyframe_embeddings.npy"
_DEFAULT_MAPPING_FILENAME = "index_mapping.json"

# A per-scene ranked list ``[(scene_id, score)]`` for one film.
_RankedList = list[tuple[int, float]]
# One film's score payload: ``(state, clip_ranked, bm25_hits, metadata_ranked)``.
# ``state`` is the Phase-4 materialisation lookup; the three lists are this
# film's per-scene contributions to the CLIP / BM25 / metadata global lists.
_FilmScore = tuple[dict[str, Any], _RankedList, _RankedList, _RankedList]
# A GLOBAL ranked list keyed by ``(film_slug, scene_id)``.
_GlobalList = list[tuple[tuple[str, int], float]]


@dataclass(frozen=True)
class _ScoringContext:
    """Loop-invariant inputs shared across every per-film scoring call.

    Built once by :func:`_collect_global_lists` and threaded into
    :func:`_score_film` so the per-film signature stays small. ``clip_scorer``
    / ``bm25_scorer`` are stateless and reused across films.
    """

    cfg: Settings
    query: str
    text_vec: np.ndarray
    min_similarity: float
    selected_tags: list[str]
    raw_k: int
    retriever_mode: str
    clip_scorer: CLIPScorer
    bm25_scorer: BM25Scorer


def _get_embedder(cfg: Settings) -> Any:
    """Return a fresh image embedder via the registry. Module-scope so unit
    tests monkeypatch ``kuaa.search.aggregate._get_embedder`` to avoid
    loading the real model.

    Dispatch honours ``cfg.models.image_embedder`` so the SigLIP-multilingual
    backend (M3 pre-flight Task 4.2 flip) returns a SigLIP encoder whose
    output dim matches the on-disk index, instead of the previous hardcoded
    ``OpenClipEmbedder()`` which 500'd on dim mismatch against a 1024-dim
    SigLIP index.
    """
    from kuaa.models.registry import get_image_embedder

    return get_image_embedder(cfg)


def _get_search_index(cfg: Settings, slug: str) -> SearchIndex:
    """Return the (cached) :class:`SearchIndex` for ``slug``. Reads
    ``cfg.embeddings.*`` filenames when present, otherwise falls back
    to the module-level defaults for minimal test configs.
    """
    emb_cfg = getattr(cfg, "embeddings", None)
    embeddings_filename = (
        getattr(emb_cfg, "filename", _DEFAULT_EMBEDDINGS_FILENAME)
        if emb_cfg is not None
        else _DEFAULT_EMBEDDINGS_FILENAME
    )
    mapping_filename = (
        getattr(emb_cfg, "mapping_filename", _DEFAULT_MAPPING_FILENAME)
        if emb_cfg is not None
        else _DEFAULT_MAPPING_FILENAME
    )
    ctx = FilmContext.for_film(cfg, slug)
    return load_index(
        ctx, embeddings_filename=embeddings_filename, mapping_filename=mapping_filename, cfg=cfg
    )


def _get_bm25_index_for_ctx_with_cfg(cfg: Settings, ctx: FilmContext) -> Any:
    """Load + cache the BM25 index for one film.

    Resolves ``cfg.search.bm25`` tunables (``stopwords_lang`` / ``k1`` /
    ``b``) directly from the supplied ``cfg`` object — no ``api.deps``
    dependency needed because aggregate callers already hold ``cfg``.
    """
    from kuaa.search.bm25 import bm25_index_for_ctx

    bm25_cfg = getattr(cfg.search, "bm25", None) if hasattr(cfg, "search") else None
    stopwords_lang = getattr(bm25_cfg, "stopwords_lang", None) if bm25_cfg else None
    k1 = float(getattr(bm25_cfg, "k1", 1.5)) if bm25_cfg else 1.5
    b = float(getattr(bm25_cfg, "b", 0.75)) if bm25_cfg else 0.75
    return bm25_index_for_ctx(
        ctx,
        stopwords_lang=stopwords_lang,
        k1=k1,
        b=b,
    )


def has_indexed_films(cfg: Settings) -> bool:
    """``True`` iff at least one registered film has an OK :class:`SearchIndex`.

    Lets the route distinguish "no indexed films yet" (run the pipeline)
    from "indexed films exist but the query matched nothing" (no results).
    """
    from kuaa.library import scan_library

    library_dir = Path(cfg.paths.library_dir)
    for film in scan_library(library_dir):
        try:
            idx = _get_search_index(cfg, film.slug)
        except ValueError:
            continue
        if idx.status is IndexStatus.OK:
            return True
    return False


@dataclass(frozen=True)
class _FilmArtifacts:
    """Per-film JSON artefacts the scoring + materialisation passes consume."""

    fps: float
    meta_by_scene: dict[Any, Any]
    descriptions: list[dict[str, Any]]
    tag_index: dict[str, Any]
    visual_rows: list[dict[str, Any]]


def _load_film_artifacts(ctx: FilmContext) -> _FilmArtifacts:
    """Load the five per-film metadata artefacts off ``ctx.metadata_dir``.

    Each loader tolerates a missing/non-list payload (``[]`` / ``{}``), exactly
    as the pre-C1 inline block did, so a film with partial metadata still scores.
    """
    kf_meta_data = load_json(ctx.metadata_dir / "keyframes_metadata.json") or []
    kf_meta = kf_meta_data if isinstance(kf_meta_data, list) else []
    descriptions_data = load_json(ctx.metadata_dir / "scene_descriptions.json") or []
    visual_data = load_json(ctx.metadata_dir / "visual_analysis.json") or []
    return _FilmArtifacts(
        fps=derive_fps(kf_meta),
        meta_by_scene={e["scene_id"]: e for e in kf_meta if "scene_id" in e},
        descriptions=descriptions_data if isinstance(descriptions_data, list) else [],
        tag_index=load_tag_index(ctx.metadata_dir) or {},
        visual_rows=visual_data if isinstance(visual_data, list) else [],
    )


def _film_bm25_hits(
    sctx: _ScoringContext,
    film_ctx: FilmContext,
    film: Any,
    allowed_scene_keys: set[str] | None,
) -> _RankedList:
    """Per-film BM25 hits, or ``[]`` for ``"clip"`` mode / an empty corpus.

    ``"clip"`` mode skips BM25 loading entirely (no disk read for a corpus
    we'll ignore). A loader failure or an unbuilt corpus contributes nothing —
    in hybrid mode the scene still surfaces via CLIP, in pure-bm25 mode it
    surfaces nothing (correct: BM25 has no signal). Verbatim from pre-C1.
    """
    if sctx.retriever_mode == "clip":
        return []
    try:
        bm25 = _get_bm25_index_for_ctx_with_cfg(sctx.cfg, film_ctx)
    except (FileNotFoundError, OSError, ValueError):
        # Narrow set of loader failure modes — anything else is a programming
        # bug we want to surface, not silently absorb.
        logger.warning(
            "aggregate_search: bm25 loader failed for %s; "
            "contributing no BM25 entries for this film",
            film.slug,
            exc_info=True,
        )
        bm25 = None
    if bm25 is None or bm25.model is None:
        logger.info(
            "aggregate_search: film=%s bm25 empty; no BM25 entries (mode=%s requested)",
            film.slug,
            sctx.retriever_mode,
        )
    return sctx.bm25_scorer.score(
        bm25=bm25,
        query=sctx.query,
        raw_k=sctx.raw_k,
        allowed_scene_keys=allowed_scene_keys,
    )


def _allowed_scene_keys(selected_tags: list[str], tag_index: dict[str, Any]) -> set[str] | None:
    """AND-intersect the selected tags into a canonical scene-key set.

    Returns ``None`` when no tags are selected (no filter), or the intersected
    membership set otherwise (possibly empty → caller skips the film). Mirrors
    ``SemanticSearch.combined``: normalise the raw tag_index, intersect sets.
    """
    if not selected_tags:
        return None
    norm_index = normalize_tag_index(tag_index)
    allowed = set(norm_index.get(selected_tags[0], set()))
    for t in selected_tags[1:]:
        allowed &= set(norm_index.get(t, set()))
    return allowed


def _score_film(sctx: _ScoringContext, cand: CandidateFilm, film: Any) -> _FilmScore | None:
    """Score one candidate film into its per-film state + three ranked lists.

    Returns ``(state, clip_ranked, bm25_hits, metadata_ranked)`` where
    ``state`` is the Phase-4 lookup payload (film + kf_df + best-row map +
    fps + scene→meta) and the three lists are this film's per-scene
    contributions (CLIP cosine, BM25, metadata-lexical). Returns ``None``
    when a tag pre-filter (AND-intersection over ``selected_tags``) excludes
    the whole film — the legacy ``continue`` that skips it entirely.

    Reads ``cand.index`` directly (load-once): the index is never re-loaded
    here. Per-film logging is emitted verbatim from the pre-C1 loop body.
    """
    idx = cand.index  # load-once: read off the candidate, never re-load
    film_ctx = FilmContext.for_film(sctx.cfg, film.slug)
    art = _load_film_artifacts(film_ctx)
    metadata_scores = (
        MetadataScorer().score(
            query=sctx.query,
            descriptions=art.descriptions,
            tag_index=art.tag_index,
            visual_rows=art.visual_rows,
        )
        if sctx.retriever_mode == "hybrid"
        else {}
    )
    metadata_ranked: _RankedList = sorted(
        metadata_scores.items(), key=lambda p: p[1], reverse=True
    )[: sctx.raw_k]

    allowed_scene_keys = _allowed_scene_keys(sctx.selected_tags, art.tag_index)
    if sctx.selected_tags:
        if not allowed_scene_keys:
            return None  # tag intersection empty → skip the whole film
        metadata_ranked = [
            (sid, s) for sid, s in metadata_ranked if scene_id_key(sid) in allowed_scene_keys
        ]

    embeddings: Any = idx.embeddings
    kf_df: Any = idx.kf_df

    # CLIP-side ranked list (best keyframe per scene, descending).
    clip_ranked, best_row_by_sid = sctx.clip_scorer.score(
        embeddings=embeddings,
        kf_df=kf_df,
        text_vec=sctx.text_vec,
        min_similarity=sctx.min_similarity,
        allowed_scene_keys=allowed_scene_keys,
        raw_k=sctx.raw_k,
    )
    bm25_hits = _film_bm25_hits(sctx, film_ctx, film, allowed_scene_keys)

    state: dict[str, Any] = {
        "film": film,
        "kf_df": kf_df,
        "best_row_by_sid": best_row_by_sid,
        "fps": art.fps,
        "meta_by_scene": art.meta_by_scene,
    }

    scores: np.ndarray = embeddings @ sctx.text_vec
    if scores.size:
        top3 = np.sort(scores)[-3:][::-1]
        logger.info(
            "aggregate_search: film=%s n_vectors=%d top3=%s "
            "clip_n=%d bm25_n=%d metadata_n=%d (retriever=%s)",
            film.slug,
            int(scores.size),
            [round(float(s), 3) for s in top3],
            len(clip_ranked),
            len(bm25_hits),
            len(metadata_ranked),
            sctx.retriever_mode,
        )
    return state, clip_ranked, bm25_hits, metadata_ranked


def _dispatch_ranked(
    retriever_mode: str,
    *,
    global_clip: _GlobalList,
    global_bm25: _GlobalList,
    global_metadata: _GlobalList,
    sem_w: float,
    bm25_w: float,
    rrf_k: int,
) -> _GlobalList:
    """Build the unified ``ranked`` list from the three global lists (Phase 3).

    ``"clip"`` / ``"bm25"`` surface their global list as-is. ``"hybrid"``
    fuses via weighted RRF: when a metadata signal exists it carries a fixed
    0.65 weight and the CLIP/BM25 residual is split by their normalised
    sem/bm25 weights; otherwise it is the plain two-way sem/bm25 RRF. All
    weighting arithmetic is verbatim from the pre-C1 Phase-3 dispatch.

    Global RRF (over the cross-film-concatenated lists) is deliberate: the
    pre-decomposition implementation ran per-film RRF and then sorted across
    films by raw RRF score, which is DEGENERATE — every film's per-film rank-1
    contribution gets the same ``1/(rrf_k+1)`` score, so the cross-film top-K
    ordering was decided by film-iteration order, not signal strength.
    Assigning each item a single GLOBAL rank per side breaks that tie.
    """
    if retriever_mode == "clip":
        return global_clip
    if retriever_mode == "bm25":
        return global_bm25
    # "hybrid"
    if global_metadata:
        metadata_w = 0.65
        residual_w = 1.0 - metadata_w
        retrieval_total = max(float(sem_w) + float(bm25_w), 1e-12)
        return _fuse_rrf_many(
            [
                (global_metadata, metadata_w),
                (global_clip, residual_w * float(sem_w) / retrieval_total),
                (global_bm25, residual_w * float(bm25_w) / retrieval_total),
            ],
            k_rrf=rrf_k,
        )
    return _fuse_rrf_many(
        [
            (global_clip, float(sem_w)),
            (global_bm25, float(bm25_w)),
        ],
        k_rrf=rrf_k,
    )


def _collect_global_lists(
    sctx: _ScoringContext,
    candidates: list[CandidateFilm],
    film_by_slug: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], _GlobalList, _GlobalList, _GlobalList]:
    """Phase 1+2: score every candidate, then sort the three global lists.

    Each :func:`_score_film` reads ``cand.index`` (load-once) and returns this
    film's tag-filtered per-scene contributions; ``None`` means the tag
    pre-filter excluded the whole film (the legacy ``continue``). The three
    GLOBAL lists are keyed by ``(film_slug, scene_id)`` and returned sorted by
    score, descending. ``per_film`` is the Phase-4 materialisation lookup.
    """
    per_film: dict[str, dict[str, Any]] = {}
    global_clip: _GlobalList = []
    global_bm25: _GlobalList = []
    global_metadata: _GlobalList = []

    for cand in candidates:
        film = film_by_slug[cand.slug]
        scored = _score_film(sctx, cand, film)
        if scored is None:
            continue
        state, clip_ranked, bm25_hits, metadata_ranked = scored
        per_film[film.slug] = state
        for sid, s in clip_ranked:
            global_clip.append(((film.slug, sid), s))
        for sid, s in bm25_hits:
            global_bm25.append(((film.slug, sid), s))
        for sid, s in metadata_ranked:
            global_metadata.append(((film.slug, sid), s))

    global_clip.sort(key=lambda p: p[1], reverse=True)
    global_bm25.sort(key=lambda p: p[1], reverse=True)
    global_metadata.sort(key=lambda p: p[1], reverse=True)
    return per_film, global_clip, global_bm25, global_metadata


def _resolve_candidates(
    cfg: Settings,
) -> tuple[list[CandidateFilm], dict[str, Any]] | None:
    """Scan the library and gate to films with an OK index (load-once).

    Returns ``(candidates, film_by_slug)`` or ``None`` when the library is
    empty or no film has an indexable CLIP index — both map to an empty
    aggregate result. The film list is materialised BEFORE the embedder so an
    empty library short-circuits without paying the ~4 s CLIP model init.

    FilmFilter loads each film's index exactly once and attaches the loaded
    :class:`SearchIndex` to its candidate, collapsing the pre-C1 double load
    (a pre-scan ``valid_slugs`` pass plus a re-loading main loop). The injected
    loader is ``_get_search_index`` — monkeypatched in tests, cached in prod —
    so test fixtures that stub the index are respected.
    """
    from kuaa.library import scan_library

    films = list(scan_library(Path(cfg.paths.library_dir)))
    if not films:
        return None
    film_by_slug = {film.slug: film for film in films}
    candidates = FilmFilter(load_index=_get_search_index).candidates(
        cfg=cfg, slugs=[film.slug for film in films]
    )
    if not candidates:
        return None
    return candidates, film_by_slug


def _encode_query_vec(cfg: Settings, query: str) -> np.ndarray:
    """Encode + L2-normalise the text query into the joint embedding space."""
    embedder = _get_embedder(cfg)
    text_vec: np.ndarray = embedder.encode_text(query)
    norm = float(np.linalg.norm(text_vec))
    return text_vec / (norm + 1e-12)


def _log_query_start(
    sctx: _ScoringContext,
    *,
    num_films: int,
    top_k: int,
    sem_w: float,
    bm25_w: float,
    rrf_k: int,
) -> None:
    """Emit the per-query INFO line (verbatim format from pre-C1)."""
    logger.info(
        "aggregate_search: query=%r films=%d top_k=%d tags=%s min_sim=%.3f "
        "retriever=%s sem_w=%.2f bm25_w=%.2f rrf_k=%d",
        sctx.query,
        num_films,
        top_k,
        sctx.selected_tags or None,
        sctx.min_similarity,
        sctx.retriever_mode,
        sem_w,
        bm25_w,
        rrf_k,
    )


def _log_result(
    query: str,
    *,
    global_clip: _GlobalList,
    global_bm25: _GlobalList,
    global_metadata: _GlobalList,
    all_hits: list[dict],
) -> None:
    """Emit the result-summary INFO line (verbatim format from pre-C1)."""
    logger.info(
        "aggregate_search: query=%r global_clip=%d global_bm25=%d global_metadata=%d "
        "returned=%d top_score=%.6f",
        query,
        len(global_clip),
        len(global_bm25),
        len(global_metadata),
        len(all_hits),
        float(all_hits[0]["score"]) if all_hits else 0.0,
    )


def aggregate_search(
    cfg: Settings,
    *,
    query: str,
    modality: str,
    top_k: int,
    tags: list[str] | None = None,
    min_similarity: float = 0.0,
    retriever_mode: str = "clip",
    sem_w: float = 0.70,
    bm25_w: float = 0.30,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[dict]:
    """Run cross-film search using GLOBAL retrieval lists, not per-film.

    Text modality only (image / audio / fusion land in later plans). Four
    phases, each in a helper: :func:`_resolve_candidates` (scan + load-once
    index gate), :func:`_collect_global_lists` (Phase 1+2 — per-film scoring
    into three GLOBAL ``(film_slug, scene_id)``-keyed lists, sorted desc),
    :func:`_dispatch_ranked` (Phase 3 — clip / bm25 / hybrid-RRF), and
    :func:`materialize_hits` (Phase 4 — top_k → hit dicts).

    ``tags`` AND-intersects across selected tags via the same
    ``normalize_tag_index`` / ``scene_id_key`` pipeline as
    ``SemanticSearch.combined`` — applied identically to the CLIP and BM25
    sides so ``?retriever=bm25&tags=outdoor`` cannot silently ignore the tag.

    ``min_similarity`` floors the CLIP cosine before it enters the global
    list. It is NOT applied to BM25 scores (different scale) or fused RRF
    scores (different scale again) — same contract as the legacy code.
    """
    if modality != "text":
        raise NotImplementedError(
            f"modality={modality!r} lands in a later plan; only 'text' is supported here"
        )

    resolved = _resolve_candidates(cfg)
    if resolved is None:
        return []
    candidates, film_by_slug = resolved

    # Widen the per-film retrieval window so the global pool stays dense
    # enough to fill ``top_k`` after fusion. Short object queries often need
    # lexical/object matches rescued from below the weak SigLIP top ranks.
    sctx = _ScoringContext(
        cfg=cfg,
        query=query,
        text_vec=_encode_query_vec(cfg, query),
        min_similarity=min_similarity,
        selected_tags=list(tags) if tags else [],
        raw_k=max(top_k * 12, 50, 1),
        retriever_mode=retriever_mode,
        clip_scorer=CLIPScorer(),
        bm25_scorer=BM25Scorer(),
    )
    _log_query_start(
        sctx, num_films=len(film_by_slug), top_k=top_k, sem_w=sem_w, bm25_w=bm25_w, rrf_k=rrf_k
    )

    # Phase 1+2: score each candidate into the three GLOBAL ranked lists
    # (keyed by ``(film_slug, scene_id)``), sorted by score descending.
    # ``per_film`` carries every object Phase 4 needs for a pure look-up.
    per_film, global_clip, global_bm25, global_metadata = _collect_global_lists(
        sctx, candidates, film_by_slug
    )

    # Phase 3: dispatch by mode. ``ranked`` is the unified output —
    # a list of ``((film_slug, scene_id), score)`` pairs, top first.
    ranked = _dispatch_ranked(
        retriever_mode,
        global_clip=global_clip,
        global_bm25=global_bm25,
        global_metadata=global_metadata,
        sem_w=sem_w,
        bm25_w=bm25_w,
        rrf_k=rrf_k,
    )

    # Phase 4: materialise hit dicts. Keys are already unique
    # ((film_slug, scene_id)) so no dedupe pass is needed.
    all_hits = materialize_hits(ranked, per_film, top_k)
    _log_result(
        query,
        global_clip=global_clip,
        global_bm25=global_bm25,
        global_metadata=global_metadata,
        all_hits=all_hits,
    )
    return all_hits


# ── Typed public wrapper (T13) ────────────────────────────────────────────────
# ``aggregate(query, *, cfg, mode, top_k, filters, weights)`` is the public
# verb. It wraps the legacy ``aggregate_search`` (above, dict-returning)
# with the locked P1 API surface — typed in, typed out. P2 replaces the
# ``cfg=`` argument with ``library=Library`` once the Library type
# lands; the rest of the signature stays.


def aggregate(
    query: Query,
    *,
    cfg: Settings,
    mode: SearchMode = "clip",
    top_k: int = 20,
    filters: Filters | None = None,
    weights: HybridWeights | None = None,
) -> SearchResult:
    """Public verb: cross-film aggregate search.

    Wraps the legacy :func:`aggregate_search` (dict-returning) with the
    typed API. P1 supports text queries only — image / audio / fusion
    modalities land in later plans. ``cfg`` is the existing app-config
    handle; P2 replaces it with ``library=Library``.

    Returns a typed :class:`SearchResult`. ``no_index=True`` carries the
    empty-library / unindexed-films signal so the caller renders the
    no-index UI state.
    """
    if query.text is None:
        raise NotImplementedError(
            "aggregate() supports text queries only in P1; "
            "image / audio / fusion modalities land in later plans."
        )
    filters = filters or Filters()
    weights = weights or HybridWeights()

    from kuaa.library import scan_library
    from kuaa.timing import timed

    num_films = len(list(scan_library(Path(cfg.paths.library_dir))))

    with timed("aggregate") as t:
        raw = aggregate_search(
            cfg,
            query=query.text,
            modality="text",
            top_k=top_k,
            tags=list(filters.tags) or None,
            min_similarity=filters.min_similarity,
            retriever_mode=mode,
            sem_w=weights.sem_w,
            bm25_w=weights.bm25_w,
            rrf_k=weights.rrf_k,
        )
    hits = [
        Hit(
            scene_id=int(h["scene_id"]),
            score=float(h["score"]),
            keyframe_path=str(h.get("keyframe_path", "")),
            film_slug=h.get("film_slug"),
            film_title=h.get("film_title"),
            timecode=h.get("timecode", ""),
        )
        for h in raw
    ]
    if hits:
        no_index = False
    else:
        no_index = not has_indexed_films(cfg)
    return SearchResult(
        hits=hits,
        mode=mode,
        weights=weights if mode == "hybrid" else None,
        query=query,
        no_index=no_index,
        fusion_used=(mode == "hybrid"),
        retriever_mode=mode,
        num_films_searched=num_films,
        latency_ms=t.elapsed_ms,
    )


def aggregate_image_search(
    cfg: Settings,
    image_path: Path | str,
    top_k: int,
) -> list[dict]:
    """Run image-similarity search across all registered films.

    Loops over every indexed film (via :func:`kuaa.library.scan_library`),
    runs per-film CLIP image search, and returns a unified top-k list sorted by
    cosine similarity.  Returns hit dicts shaped like
    :func:`aggregate_hits_to_template_dicts` output so the same template renders.

    Films whose index is missing or corrupt are silently skipped (a warning is
    logged).  When the library has no indexed films the function returns ``[]``
    rather than falling back to the legacy ``data/embeddings/`` flat index.
    """
    from kuaa.library import FilmContext, keyframe_url, scan_library
    from kuaa.search.clip import search_image as _search_image

    library_dir = Path(cfg.paths.library_dir)
    data_dir = Path(cfg.paths.data_dir).resolve()
    emb_file = getattr(getattr(cfg, "embeddings", None), "filename", _DEFAULT_EMBEDDINGS_FILENAME)
    map_file = getattr(
        getattr(cfg, "embeddings", None), "mapping_filename", _DEFAULT_MAPPING_FILENAME
    )
    all_hits: list[dict] = []

    for film in scan_library(library_dir):
        try:
            film_ctx = FilmContext.for_film(cfg, film.slug)
            index = load_index(
                film_ctx, embeddings_filename=emb_file, mapping_filename=map_file, cfg=cfg
            )
            if not index.ok:
                continue
            df = _search_image(index, image_path, top_k)
            for row in df.to_dict("records"):
                all_hits.append(
                    {
                        "film_slug": film.slug,
                        "film_title": str(getattr(film, "title", film.slug)),
                        "scene_id": row["scene_id"],
                        "similarity": float(row["similarity"]),
                        "img_url": keyframe_url(str(row["filepath"]), data_dir),
                        "description": str(row.get("description", "")),
                        "timecode": "",
                    }
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("aggregate_image_search: film %r skipped: %s", film.slug, exc)

    all_hits.sort(key=lambda h: h["similarity"], reverse=True)
    return all_hits[:top_k]


def aggregate_hits_to_template_dicts(cfg: Settings, hits: list[dict]) -> list[dict]:
    """Convert ``aggregate_search`` raw hits to ``.b-card``-shaped template dicts.

    Relocated from the aggregate path of ``api/routes/search.py`` (T15).
    The ``data_dir`` MUST be the media-mount root (``cfg.paths.data_dir``),
    not ``library_dir`` — otherwise ``keyframe_url``'s ``relative_to``
    check fails for filepaths stored under ``data/frames/...`` or
    ``data/library/<slug>/...`` and the template gets
    ``img_url=None`` for every row.

    Aggregate hits already carry ``film_slug`` / ``film_title`` /
    ``timecode`` (computed by ``aggregate_search``). Template uses
    ``r.similarity``; aggregate_search emits ``score`` (cosine over the
    per-film index) — aliased here so the same ``partials/search_results.html``
    works for the per-film and aggregate paths.
    """
    from kuaa.library import keyframe_url

    data_dir = Path(cfg.paths.data_dir).resolve()
    return [
        {
            "film_slug": h["film_slug"],
            "film_title": h["film_title"],
            "scene_id": h["scene_id"],
            "similarity": h["score"],
            "img_url": keyframe_url(h["keyframe_path"], data_dir),
            "timecode": h["timecode"],
        }
        for h in hits
    ]
