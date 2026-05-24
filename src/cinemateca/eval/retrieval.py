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

from cinemateca.eval.datasets import EvaluationDataset
from cinemateca.eval.metrics import RetrievalResult, evaluate_query, summarize_results
from cinemateca.retrieval.hybrid import DEFAULT_RRF_K, fuse_rrf, resolve_weights
from cinemateca.scene_ids import scene_id_key

logger = logging.getLogger(__name__)


VALID_RETRIEVERS = ("clip", "bm25", "hybrid")


class EvalError(RuntimeError):
    """Raised for clear user-facing evaluation failures."""


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
) -> list[dict[str, Any]]:
    """BM25-only ranking. Surfaces best keyframe per scene from ``kf_df``."""
    hits = bm25.query(query_text, top_k=raw_k)
    if not hits:
        return []
    # Build a quick first-keyframe lookup so BM25-only scenes carry a
    # filepath in top_results (matches ``_pick_kf_rows_by_sid`` fallback —
    # no per-query CLIP encode needed for "best frame" in eval).
    first_row_by_sid: dict[int, dict[str, Any]] = {}
    if kf_df is not None and len(kf_df) > 0:
        for r in kf_df.itertuples(index=False):
            sid = int(getattr(r, "scene_id", -1))
            if sid in first_row_by_sid or sid < 0:
                continue
            first_row_by_sid[sid] = {
                "filepath": str(getattr(r, "filepath", "")),
            }
    rows = []
    for sid, score in hits:
        rows.append(
            {
                "scene_id": sid,
                "similarity": float(score),
                "filepath": first_row_by_sid.get(int(sid), {}).get("filepath", ""),
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
    clip_row_by_sid: dict[int, dict[str, Any]] = {
        int(r["scene_id"]): r for r in clip_rows
    }
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

    embeddings, mapping, kf_df, emb_path, map_path = _load_clip_index(cfg)

    from cinemateca.device import device_from_config
    from cinemateca.embeddings import SemanticSearch
    from cinemateca.models.base import ImageEmbedder
    from cinemateca.models.registry import get_image_embedder

    # The embedder is needed for CLIP and hybrid. We build it for BM25 too
    # because the regression path may need text encoding for backfill if we
    # extend later — cheap to construct here, and keeps the harness symmetric.
    embedder: ImageEmbedder = get_image_embedder(cfg, device_from_config(cfg))
    searcher = SemanticSearch(embeddings, kf_df, embedder)
    index_scene_ids = tuple(scene_id_key(v) for v in kf_df["scene_id"].tolist())

    bm25 = None
    bm25_corpus_size = 0
    if retriever in ("bm25", "hybrid"):
        metadata_dir = Path(cfg.paths.metadata_dir)
        bm25 = _build_bm25_index(cfg, metadata_dir)
        bm25_corpus_size = len(bm25.scene_ids)
        if bm25.model is None:
            raise EvalError(
                f"BM25 corpus is empty under {metadata_dir} — scene_descriptions.json "
                "and scene_tags.json yield no usable documents"
            )

    if retriever == "hybrid":
        sem_w, bm25_w = resolve_weights(sem_w=sem_w, bm25_w=bm25_w, defaults=(0.5, 0.5))

    # Pull a wider candidate set than top_k so post-dedup we still have
    # `top_k` distinct scenes (mirrors `search_text` 4× widening).
    raw_k = max(top_k * 4, 10)

    query_results: list[RetrievalResult] = []
    warnings: list[str] = []
    for query in dataset.queries:
        if retriever == "clip":
            ranked_rows = _clip_rank(searcher, kf_df, query.text, raw_k=raw_k)
        elif retriever == "bm25":
            ranked_rows = _bm25_rank(bm25, kf_df, query.text, raw_k=raw_k)
        else:
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
        "embeddings_path": str(emb_path),
        "mapping_path": str(map_path),
        "model": mapping.get("model", f"{cfg.embeddings.model} ({cfg.embeddings.pretrained})"),
        "dimension": mapping.get("dimension"),
        "total_vectors": len(kf_df),
        "top_k": top_k,
        "retriever": retriever,
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
