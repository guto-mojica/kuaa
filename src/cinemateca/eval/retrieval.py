"""Run retrieval evaluation against a stored CLIP and/or BM25 index.

Supports three retriever modes:

  * ``clip`` — the original semantic search via ``SemanticSearch.by_text``.
  * ``bm25`` — pure BM25 over Moondream descriptions + merged tag index.
  * ``hybrid`` — weighted RRF fusion of the above (mirrors ``search_hybrid``
    in ``api/services/search.py``). Defaults to the same RRF constant the
    web UI uses (``cinemateca.retrieval.hybrid.DEFAULT_RRF_K``).

The harness dedupes ranked output by ``scene_id`` keeping the first
occurrence (= highest similarity / fused score) so every mode is compared
on a *scene-level* ranking — matching the UI contract and avoiding
duplicate-keyframe inflation of nDCG.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cinemateca.errors import EvalError
from cinemateca.eval.datasets import EvaluationDataset
from cinemateca.eval.metrics import RetrievalResult, evaluate_query, summarize_results
from cinemateca.eval.slates import ModalQuery, generate_slate
from cinemateca.reproducibility import seed_everything
from cinemateca.retrieval.hybrid import DEFAULT_RRF_K, fuse_rrf, resolve_weights
from cinemateca.rhymes.algorithm import _SCENE_NUM_RE
from cinemateca.scene_ids import scene_id_key

logger = logging.getLogger(__name__)


VALID_RETRIEVERS = ("clip", "bm25", "hybrid")


@dataclass(frozen=True)
class RetrievalRun:
    """Complete retrieval evaluation result."""

    dataset: EvaluationDataset
    metrics: dict[str, float | int]
    query_results: tuple[RetrievalResult, ...]
    context: dict[str, Any]
    warnings: tuple[str, ...] = field(default_factory=tuple)


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise EvalError(f"{label} not found: {path}")


def _result_rows(rows_in: list[dict[str, Any]], *, limit: int) -> tuple[dict[str, Any], ...]:
    out: list[dict[str, Any]] = []
    for row in rows_in[:limit]:
        out.append(
            {
                "rank": int(row.get("rank", len(out) + 1)),
                "scene_id": scene_id_key(row.get("scene_id", "")),
                "similarity": float(row.get("similarity", 0.0)),
                "filepath": str(row.get("filepath", "")),
            }
        )
    return tuple(out)


def _dedup_by_scene(
    ranked: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep the first occurrence of each ``scene_id`` (= best score).

    Inputs are expected ordered descending by similarity/fused-score.
    The dedup mirrors what ``api.services.search.search_text`` does so the
    eval ranks scenes, not keyframes.
    """
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in ranked:
        sid = scene_id_key(row.get("scene_id", ""))
        if sid in seen:
            continue
        seen.add(sid)
        out.append(row)
    # Renumber ranks after dedup so ``top_results`` carries a contiguous rank.
    for i, row in enumerate(out, start=1):
        row["rank"] = i
    return out


def _load_clip_index(cfg) -> tuple[Any, dict, Any, Path, Path]:
    """Return ``(embeddings, mapping, kf_df, emb_path, map_path)``."""
    emb_path = Path(cfg.paths.embeddings_dir) / cfg.embeddings.filename
    map_path = Path(cfg.paths.embeddings_dir) / cfg.embeddings.mapping_filename
    _require_file(emb_path, "Embeddings file")
    _require_file(map_path, "Index mapping file")

    from cinemateca.models.clip.openclip import OpenClipEmbedder

    try:
        embeddings, mapping, kf_df = OpenClipEmbedder.load(emb_path, map_path)
    except Exception as exc:
        raise EvalError(f"Could not load search index: {exc}") from exc

    n_emb = int(getattr(embeddings, "shape", [0])[0])
    n_map = len(kf_df)
    declared = mapping.get("total_vectors")
    if n_emb != n_map:
        raise EvalError(f"Search index row mismatch: {n_emb} embeddings vs {n_map} mapped rows")
    if declared is not None and int(declared) != n_map:
        raise EvalError(
            f"Search index declares total_vectors={declared} but has {n_map} mapped rows"
        )
    return embeddings, mapping, kf_df, emb_path, map_path


def _load_descriptions(metadata_dir: Path) -> list[dict]:
    path = metadata_dir / "scene_descriptions.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning("eval: malformed %s — using empty descriptions", path)
        return []
    return data if isinstance(data, list) else []


def _load_tag_index(metadata_dir: Path) -> dict[str, list[int]]:
    """Merged ``{tag: [scene_id, ...]}`` (LLM tags + manual annotations).

    Replicates ``api/services/catalog.load_tag_index`` without importing the
    HTTP layer — the eval harness has no HTTP dependency.
    """
    out: dict[str, list[int]] = {}
    tags_path = metadata_dir / "scene_tags.json"
    if tags_path.exists():
        try:
            raw = json.loads(tags_path.read_text())
            if isinstance(raw, dict):
                for tag, sids in raw.items():
                    if not isinstance(sids, list):
                        continue
                    bucket = out.setdefault(str(tag), [])
                    for sid in sids:
                        try:
                            bucket.append(int(sid))
                        except (TypeError, ValueError):
                            continue
        except json.JSONDecodeError:
            logger.warning("eval: malformed %s — skipping LLM tags", tags_path)

    annot_path = metadata_dir / "manual_annotations.json"
    if annot_path.exists():
        try:
            raw = json.loads(annot_path.read_text())
            if isinstance(raw, dict):
                # Manual annotations are scene_id-keyed dicts. Each value
                # is a list of tag strings; merge them into the tag-keyed
                # inverted index.
                for sid_str, tags in raw.items():
                    if not isinstance(tags, list):
                        continue
                    try:
                        sid_int = int(sid_str)
                    except (TypeError, ValueError):
                        continue
                    for tag in tags:
                        out.setdefault(str(tag), []).append(sid_int)
        except json.JSONDecodeError:
            logger.warning("eval: malformed %s — skipping manual tags", annot_path)
    return out


def _build_bm25_index(cfg, metadata_dir: Path):
    """Construct a per-film ``BM25Index`` from on-disk artefacts.

    Tries to honour ``cfg.search.bm25`` if present (k1, b, stopwords_lang);
    falls back to BM25Okapi defaults.
    """
    from cinemateca.retrieval.bm25 import BM25Index

    bm25_cfg = getattr(getattr(cfg, "search", None), "bm25", None)
    k1 = float(getattr(bm25_cfg, "k1", 1.5)) if bm25_cfg else 1.5
    b = float(getattr(bm25_cfg, "b", 0.75)) if bm25_cfg else 0.75
    stopwords_lang = getattr(bm25_cfg, "stopwords_lang", None) if bm25_cfg else None

    descriptions = _load_descriptions(metadata_dir)
    tag_index = _load_tag_index(metadata_dir)
    return BM25Index.build(
        descriptions=descriptions,
        tag_index=tag_index,
        stopwords_lang=stopwords_lang,
        k1=k1,
        b=b,
    )


def _first_keyframe_filepaths_from_metadata(metadata_dir: Path) -> dict[int, str]:
    """Return ``scene_id -> first keyframe filepath`` from metadata JSON.

    BM25-only evaluation should not require a CLIP embedding matrix just to
    populate ``top_results[].filepath``. This reads the same lightweight
    keyframe metadata file the web layer uses when available.
    """
    path = metadata_dir / "keyframes_metadata.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning("eval: malformed %s — top_results filepaths omitted", path)
        return {}
    if not isinstance(data, list):
        return {}

    out: dict[int, str] = {}
    for row in data:
        if not isinstance(row, dict):
            continue
        try:
            sid = int(row.get("scene_id", -1))
        except (TypeError, ValueError):
            continue
        if sid < 0 or sid in out:
            continue
        filepath = str(row.get("filepath") or row.get("keyframe_path") or "")
        if filepath:
            out[sid] = filepath
    return out


def _first_keyframe_filepaths_from_df(kf_df) -> dict[int, str]:
    """Return ``scene_id -> first keyframe filepath`` from an index DataFrame."""
    out: dict[int, str] = {}
    if kf_df is None or len(kf_df) == 0:
        return out
    for row in kf_df.itertuples(index=False):
        try:
            sid = int(getattr(row, "scene_id", -1))
        except (TypeError, ValueError):
            continue
        if sid < 0 or sid in out:
            continue
        out[sid] = str(getattr(row, "filepath", ""))
    return out


def _clip_rank(
    searcher,
    kf_df,
    query_text: str,
    *,
    raw_k: int,
) -> list[dict[str, Any]]:
    """CLIP-only ranking. Returns a list of dicts (scene-deduped)."""
    df = searcher.by_text(query_text, top_k=raw_k)
    rows: list[dict[str, Any]] = []
    if df.empty:
        return rows
    for r in df.itertuples(index=False):
        rows.append(
            {
                "scene_id": getattr(r, "scene_id", ""),
                "similarity": float(getattr(r, "similarity", 0.0)),
                "filepath": str(getattr(r, "filepath", "")),
            }
        )
    return _dedup_by_scene(rows)


def _bm25_rank(
    bm25,
    kf_df,
    query_text: str,
    *,
    raw_k: int,
    first_filepath_by_sid: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    """BM25-only ranking. Surfaces best keyframe per scene from ``kf_df``."""
    hits = bm25.query(query_text, top_k=raw_k)
    if not hits:
        return []
    filepath_by_sid = first_filepath_by_sid or _first_keyframe_filepaths_from_df(kf_df)
    rows = []
    for sid, score in hits:
        rows.append(
            {
                "scene_id": sid,
                "similarity": float(score),
                "filepath": filepath_by_sid.get(int(sid), ""),
            }
        )
    return _dedup_by_scene(rows)


def _hybrid_rank(
    searcher,
    bm25,
    kf_df,
    query_text: str,
    *,
    raw_k: int,
    sem_w: float,
    bm25_w: float,
    k_rrf: int,
) -> list[dict[str, Any]]:
    """RRF fusion of CLIP + BM25, then scene-deduped."""
    clip_rows = _clip_rank(searcher, kf_df, query_text, raw_k=raw_k)
    bm25_hits = bm25.query(query_text, top_k=raw_k) if bm25.model is not None else []

    clip_pairs = [(int(r["scene_id"]), float(r["similarity"])) for r in clip_rows]
    fused = fuse_rrf(clip_pairs, bm25_hits, sem_w=sem_w, bm25_w=bm25_w, k_rrf=k_rrf)

    # Build a sid -> clip_row lookup so fused rows inherit filepath when CLIP
    # surfaced them; otherwise fall back to first-keyframe per sid.
    clip_row_by_sid: dict[int, dict[str, Any]] = {int(r["scene_id"]): r for r in clip_rows}
    first_row_by_sid: dict[int, str] = {}
    if kf_df is not None and len(kf_df) > 0:
        for r in kf_df.itertuples(index=False):
            sid = int(getattr(r, "scene_id", -1))
            if sid < 0 or sid in first_row_by_sid:
                continue
            first_row_by_sid[sid] = str(getattr(r, "filepath", ""))

    rows: list[dict[str, Any]] = []
    for sid, score in fused:
        clip_row = clip_row_by_sid.get(int(sid))
        filepath = (clip_row or {}).get("filepath") or first_row_by_sid.get(int(sid), "")
        rows.append(
            {
                "scene_id": sid,
                "similarity": float(score),
                "filepath": filepath,
            }
        )
    return _dedup_by_scene(rows)


def run_retrieval_eval(
    cfg,
    dataset: EvaluationDataset,
    *,
    config_path: str | Path | None,
    top_k: int = 10,
    retriever: str = "clip",
    sem_w: float = 0.5,
    bm25_w: float = 0.5,
    k_rrf: int = DEFAULT_RRF_K,
    seed: int = 0,
) -> RetrievalRun:
    """Evaluate text queries against one of three retrievers.

    Args:
        retriever: ``"clip"``, ``"bm25"``, or ``"hybrid"``.
        sem_w, bm25_w: only consulted when ``retriever == "hybrid"``.
            Clamped to ``[0, 1]``; if both are 0, falls back to (0.5, 0.5).
        k_rrf: Reciprocal Rank Fusion rank-shift constant. Defaults to
            ``cinemateca.retrieval.hybrid.DEFAULT_RRF_K`` (60).
    """

    if top_k < 1:
        raise EvalError("top_k must be at least 1")
    if retriever not in VALID_RETRIEVERS:
        raise EvalError(f"retriever must be one of {VALID_RETRIEVERS}, got {retriever!r}")

    seed_everything(seed)

    metadata_dir = Path(cfg.paths.metadata_dir)
    embeddings = mapping = kf_df = emb_path = map_path = None
    searcher = None
    index_scene_ids: tuple[str, ...] = ()

    if retriever in ("clip", "hybrid"):
        embeddings, mapping, kf_df, emb_path, map_path = _load_clip_index(cfg)

        from cinemateca.device import device_from_config
        from cinemateca.embeddings import SemanticSearch
        from cinemateca.models.base import ImageEmbedder
        from cinemateca.models.registry import get_image_embedder

        embedder: ImageEmbedder = get_image_embedder(cfg, device_from_config(cfg))
        searcher = SemanticSearch(embeddings, kf_df, embedder)
        index_scene_ids = tuple(scene_id_key(v) for v in kf_df["scene_id"].tolist())

    bm25 = None
    bm25_corpus_size = 0
    first_filepath_by_sid: dict[int, str] | None = None
    if retriever in ("bm25", "hybrid"):
        bm25 = _build_bm25_index(cfg, metadata_dir)
        bm25_corpus_size = len(bm25.scene_ids)
        if bm25.model is None:
            raise EvalError(
                f"BM25 corpus is empty under {metadata_dir} — scene_descriptions.json "
                "and scene_tags.json yield no usable documents"
            )
        if not index_scene_ids:
            index_scene_ids = tuple(scene_id_key(v) for v in bm25.scene_ids)
        if retriever == "bm25":
            first_filepath_by_sid = _first_keyframe_filepaths_from_metadata(metadata_dir)

    if retriever == "hybrid":
        sem_w, bm25_w = resolve_weights(sem_w=sem_w, bm25_w=bm25_w, defaults=(0.5, 0.5))

    # Pull a wider candidate set than top_k so post-dedup we still have
    # `top_k` distinct scenes (mirrors `search_text` 4× widening).
    raw_k = max(top_k * 4, 10)

    query_results: list[RetrievalResult] = []
    warnings: list[str] = []
    for query in dataset.queries:
        if retriever == "clip":
            assert searcher is not None and kf_df is not None
            ranked_rows = _clip_rank(searcher, kf_df, query.text, raw_k=raw_k)
        elif retriever == "bm25":
            ranked_rows = _bm25_rank(
                bm25,
                kf_df,
                query.text,
                raw_k=raw_k,
                first_filepath_by_sid=first_filepath_by_sid,
            )
        else:
            assert searcher is not None and kf_df is not None
            ranked_rows = _hybrid_rank(
                searcher,
                bm25,
                kf_df,
                query.text,
                raw_k=raw_k,
                sem_w=sem_w,
                bm25_w=bm25_w,
                k_rrf=k_rrf,
            )
        ranked = tuple(scene_id_key(row["scene_id"]) for row in ranked_rows)
        result = evaluate_query(
            query_id=query.id,
            text=query.text,
            relevant_scene_ids=query.relevant_scene_ids,
            ranked_scene_ids=ranked,
            relevance=query.relevance,
            top_results=_result_rows(ranked_rows, limit=top_k),
            index_scene_ids=index_scene_ids,
        )
        if result.missing_relevant_scene_ids:
            missing = ", ".join(result.missing_relevant_scene_ids)
            warnings.append(f"{query.id}: relevant scene ids missing from index: {missing}")
        query_results.append(result)

    context = {
        "config_path": str(config_path) if config_path else "default",
        "queries_path": str(dataset.path) if dataset.path else "",
        "embeddings_path": str(emb_path) if emb_path else "",
        "mapping_path": str(map_path) if map_path else "",
        "model": (
            mapping.get("model", f"{cfg.embeddings.model} ({cfg.embeddings.pretrained})")
            if mapping
            else "BM25"
        ),
        "dimension": mapping.get("dimension") if mapping else None,
        "total_vectors": len(kf_df) if kf_df is not None else 0,
        "top_k": top_k,
        "retriever": retriever,
        "seed": seed,
    }
    if retriever in ("bm25", "hybrid"):
        context["bm25_corpus_size"] = bm25_corpus_size
    if retriever == "hybrid":
        context["sem_w"] = sem_w
        context["bm25_w"] = bm25_w
        context["k_rrf"] = k_rrf

    return RetrievalRun(
        dataset=dataset,
        metrics=summarize_results(query_results),
        query_results=tuple(query_results),
        context=context,
        warnings=tuple(warnings),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-modality scorers (E3b): image / audio / fusion / rhyme.
#
# Each calls cinemateca.eval.slates.generate_slate — the REAL retrieval backend
# for that modality — and scores the returned candidate rows with the same
# evaluate_query / summarize_results math the text path uses, so the result is a
# RetrievalRun the existing report writer (report.build_payload) serialises
# unchanged. The text path (run_retrieval_eval above) is deliberately untouched.
# ─────────────────────────────────────────────────────────────────────────────

# Modal scorers build a minimal EvaluationDataset whose `queries` is empty:
# report.build_payload reads only dataset.{dataset,version,source,label_status},
# not dataset.queries, so an empty tuple satisfies the writer for non-text modes
# (the loader-built dataset with QueryCase rows is the text path's concern only).


def _modal_dataset(modality: str, queries: list[ModalQuery], queries_path: Path | None):
    """Construct the minimal EvaluationDataset report.build_payload needs.

    build_payload touches ``dataset.{dataset,version,source,label_status}`` and
    iterates ``run.query_results`` (NOT ``dataset.queries``), so the empty
    ``queries=()`` tuple here is sufficient — the per-query report rows come
    from the scored ``RetrievalResult`` list, not from the dataset object.
    """
    return EvaluationDataset(
        dataset=f"m3_{modality}",
        version=1,
        queries=(),
        source={"modality": modality, "query_count": len(queries)},
        label_status="seed_curator_grading_pending",
        path=queries_path,
    )


def _default_relevance(
    query: ModalQuery, rows: list[dict[str, Any]]
) -> tuple[tuple[str, ...], dict[str, float], str]:
    """Minimal, honest relevance resolution for one modal query (E3b only).

    Returns ``(relevant_scene_ids, relevance_map, method)``. ``method`` is one
    of ``"hypothesis" | "known_item" | "pseudo"`` and is recorded in the run
    context for honesty. The full KI/PR/HY proxy labeller is task **E2**
    (``cinemateca.eval.proxy``); ``run_<modality>_eval`` accepts a
    ``relevance_resolver`` with this exact 3-tuple signature so E2 swaps its
    labeller in without touching the scorers.

    Strategy per modality:

      * **text / fusion WITH YAML hypotheses** — use the maintainer's
        ``relevant_scene_ids`` + ``relevance`` from the query file.
      * **image** — known-item: the anchor scene id parsed out of the
        ``image_path`` basename via :data:`_SCENE_NUM_RE` (reused from
        ``cinemateca.rhymes.algorithm``).
      * **rhyme** — known-item: the anchor scene id from ``"<slug>/<scene_id>"``.
        Note the rhyme slate is cross-film-only, so the anchor scene (in the
        anchor film) is excluded from candidates — this KI is intentionally
        a hard target that usually scores 0; it exists to make the GATE
        produce metrics, and E2's proxy supersedes it.
      * **audio / fusion WITHOUT YAML labels** — pseudo-relevance placeholder
        (top-1 returned scene treated as relevant).
    """
    if query.relevant_scene_ids:
        # text (always labelled) + any fusion query the maintainer labelled.
        rel = tuple(scene_id_key(s) for s in query.relevant_scene_ids)
        rmap = (
            {scene_id_key(k): float(v) for k, v in query.relevance.items()}
            if query.relevance
            else {sid: 1.0 for sid in rel}
        )
        return rel, rmap, "hypothesis"

    if query.query_type == "image" and query.image_path is not None:
        match = _SCENE_NUM_RE.search(query.image_path.name)
        if match:
            sid = scene_id_key(int(match.group(1)))
            return (sid,), {sid: 1.0}, "known_item"

    if query.query_type == "rhyme" and query.anchor and query.anchor.count("/") == 1:
        _slug, sid_s = query.anchor.split("/", 1)
        sid = scene_id_key(int(sid_s))
        return (sid,), {sid: 1.0}, "known_item"

    # audio / fusion-without-labels (and any image whose basename didn't parse):
    # PR placeholder — E2's proxy.proxy_labels (KI/PR/HY) supersedes this; here
    # only to make run_eval produce metrics for all 5 modalities (GATE).
    if rows:
        sid = scene_id_key(rows[0]["scene_id"])
        return (sid,), {sid: 1.0}, "pseudo"
    return (), {}, "pseudo"


def _modal_top_results(rows: list[dict[str, Any]], *, limit: int) -> tuple[dict[str, Any], ...]:
    """Adapt slate candidate rows to the ``top_results`` shape report writers read.

    The slate row carries ``scene_id`` / ``score`` / ``keyframe_url`` (no
    ``similarity`` / ``filepath`` / ``rank`` keys), so we map those across to
    match ``_result_rows``'s output contract used by the text path.
    """
    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows[:limit], start=1):
        out.append(
            {
                "rank": i,
                "scene_id": scene_id_key(row.get("scene_id", "")),
                "similarity": float(row.get("score", 0.0)),
                "filepath": str(row.get("keyframe_url", "")),
            }
        )
    return tuple(out)


def _run_modal_eval(
    cfg,
    queries: list[ModalQuery],
    *,
    modality: str,
    library_dir: Path,
    film_slug: str | None,
    seed: int = 0,
    top_k: int = 10,
    relevance_resolver=None,
) -> RetrievalRun:
    """Shared scoring loop for the image / audio / fusion / rhyme modalities.

    Generates each query's slate via :func:`cinemateca.eval.slates.generate_slate`
    (the real backend), resolves relevance via ``relevance_resolver`` (defaults
    to :func:`_default_relevance`), and scores with :func:`evaluate_query`.
    Returns a :class:`RetrievalRun` the report writer serialises unchanged.

    ``library_dir`` MUST be the full library root (``cfg.paths.library_dir``),
    not a single-film override — ``generate_slate`` (and ``find_rhymes``) walk
    every film under it. Single-film modalities (image/audio/fusion) are scoped
    to ``film_slug`` after generation when one is given; rhyme is inherently
    cross-film and is never scoped out.
    """
    if top_k < 1:
        raise EvalError("top_k must be at least 1")

    seed_everything(seed)
    resolve = relevance_resolver or _default_relevance

    scope_slug = film_slug if modality != "rhyme" else None
    methods: set[str] = set()
    results: list[RetrievalResult] = []
    warnings: list[str] = []

    for q in (q for q in queries if q.query_type == modality):
        rows = generate_slate(query=q, cfg=cfg, library_dir=library_dir, k=top_k)
        if scope_slug:
            rows = [r for r in rows if r.get("film_slug") == scope_slug]
        rel_ids, rel_map, method = resolve(q, rows)
        if not rel_ids:
            # No ground truth and no rows to derive a pseudo-label from →
            # cannot score this query; skip it (summarize_results needs ≥1).
            warnings.append(f"{q.id}: no candidates and no labels — query skipped")
            continue
        methods.add(method)
        ranked_ids = tuple(scene_id_key(r["scene_id"]) for r in rows)
        result = evaluate_query(
            query_id=q.id,
            text=q.text or (q.anchor or ""),
            relevant_scene_ids=rel_ids,
            ranked_scene_ids=ranked_ids,
            relevance=rel_map,
            top_results=_modal_top_results(rows, limit=top_k),
        )
        results.append(result)

    if not results:
        raise EvalError(
            f"no '{modality}' queries produced a scorable slate "
            f"(checked {sum(1 for q in queries if q.query_type == modality)} candidates)"
        )

    context: dict[str, Any] = {
        "config_path": "default",
        "queries_path": "",
        "model": modality,
        "top_k": top_k,
        "retriever": modality,
        "modality": modality,
        "film_slug": film_slug,
        "seed": seed,
        "relevance_method": "+".join(sorted(methods)) if methods else "none",
    }

    return RetrievalRun(
        dataset=_modal_dataset(modality, queries, None),
        metrics=summarize_results(results),
        query_results=tuple(results),
        context=context,
        warnings=tuple(warnings),
    )


def run_image_eval(
    cfg,
    queries: list[ModalQuery],
    *,
    library_dir: Path,
    film_slug: str | None,
    seed: int = 0,
    top_k: int = 10,
    relevance_resolver=None,
) -> RetrievalRun:
    """Score the image queries via CLIP ``find`` slates (known-item relevance)."""
    return _run_modal_eval(
        cfg,
        queries,
        modality="image",
        library_dir=library_dir,
        film_slug=film_slug,
        seed=seed,
        top_k=top_k,
        relevance_resolver=relevance_resolver,
    )


def run_audio_eval(
    cfg,
    queries: list[ModalQuery],
    *,
    library_dir: Path,
    film_slug: str | None,
    seed: int = 0,
    top_k: int = 10,
    relevance_resolver=None,
) -> RetrievalRun:
    """Score the audio queries via CLAP ``search_audio`` slates (pseudo relevance)."""
    return _run_modal_eval(
        cfg,
        queries,
        modality="audio",
        library_dir=library_dir,
        film_slug=film_slug,
        seed=seed,
        top_k=top_k,
        relevance_resolver=relevance_resolver,
    )


def run_fusion_eval(
    cfg,
    queries: list[ModalQuery],
    *,
    library_dir: Path,
    film_slug: str | None,
    seed: int = 0,
    top_k: int = 10,
    relevance_resolver=None,
) -> RetrievalRun:
    """Score the fusion queries via CLIP×CLAP ``search_fusion`` slates.

    Uses YAML hypotheses when the query carries them, else a pseudo-relevance
    placeholder (the m3_full fusion queries are unlabelled today).
    """
    return _run_modal_eval(
        cfg,
        queries,
        modality="fusion",
        library_dir=library_dir,
        film_slug=film_slug,
        seed=seed,
        top_k=top_k,
        relevance_resolver=relevance_resolver,
    )


def run_rhyme_eval(
    cfg,
    queries: list[ModalQuery],
    *,
    library_dir: Path,
    film_slug: str | None,
    seed: int = 0,
    top_k: int = 10,
    relevance_resolver=None,
) -> RetrievalRun:
    """Score the rhyme queries via cross-film ``find_rhymes`` slates.

    Rhyme is inherently cross-film (anchor film excluded from the candidate
    pool), so the scorer never scopes rows to ``film_slug``. The known-item
    relevance target is the anchor scene — intentionally a hard target the
    cross-film slate usually cannot contain (see :func:`_default_relevance`).
    """
    return _run_modal_eval(
        cfg,
        queries,
        modality="rhyme",
        library_dir=library_dir,
        film_slug=film_slug,
        seed=seed,
        top_k=top_k,
        relevance_resolver=relevance_resolver,
    )
