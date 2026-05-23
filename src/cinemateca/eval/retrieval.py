"""Run retrieval evaluation against a stored CLIP index."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cinemateca.eval.datasets import EvaluationDataset
from cinemateca.eval.metrics import RetrievalResult, evaluate_query, summarize_results
from cinemateca.scene_ids import scene_id_key


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


def _result_rows(df, *, limit: int) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for row in df.head(limit).to_dict("records"):
        rows.append(
            {
                "rank": int(row.get("rank", len(rows) + 1)),
                "scene_id": scene_id_key(row.get("scene_id", "")),
                "similarity": float(row.get("similarity", 0.0)),
                "filepath": str(row.get("filepath", "")),
            }
        )
    return tuple(rows)


def run_retrieval_eval(
    cfg,
    dataset: EvaluationDataset,
    *,
    config_path: str | Path | None,
    top_k: int = 10,
) -> RetrievalRun:
    """Evaluate text queries with the same semantic-search core as the web UI."""

    if top_k < 1:
        raise EvalError("top_k must be at least 1")

    emb_path = Path(cfg.paths.embeddings_dir) / cfg.embeddings.filename
    map_path = Path(cfg.paths.embeddings_dir) / cfg.embeddings.mapping_filename
    _require_file(emb_path, "Embeddings file")
    _require_file(map_path, "Index mapping file")

    from cinemateca.device import device_from_config
    from cinemateca.embeddings import SemanticSearch
    from cinemateca.models.base import ImageEmbedder  # noqa: F401 — type annotation
    from cinemateca.models.clip.openclip import OpenClipEmbedder
    from cinemateca.models.registry import get_image_embedder

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

    embedder: ImageEmbedder = get_image_embedder(cfg, device_from_config(cfg))
    searcher = SemanticSearch(embeddings, kf_df, embedder)
    index_scene_ids = tuple(scene_id_key(v) for v in kf_df["scene_id"].tolist())

    query_results: list[RetrievalResult] = []
    warnings: list[str] = []
    for query in dataset.queries:
        df = searcher.by_text(query.text, top_k=max(top_k, 10))
        ranked = tuple(scene_id_key(v) for v in df["scene_id"].tolist())
        result = evaluate_query(
            query_id=query.id,
            text=query.text,
            relevant_scene_ids=query.relevant_scene_ids,
            ranked_scene_ids=ranked,
            relevance=query.relevance,
            top_results=_result_rows(df, limit=top_k),
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
        "total_vectors": n_map,
        "top_k": top_k,
    }

    return RetrievalRun(
        dataset=dataset,
        metrics=summarize_results(query_results),
        query_results=tuple(query_results),
        context=context,
        warnings=tuple(warnings),
    )
