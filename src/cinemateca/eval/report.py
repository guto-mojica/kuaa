"""JSON and Markdown report rendering for evaluation runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cinemateca.eval.retrieval import RetrievalRun


def _round(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    return value


def build_payload(
    run: RetrievalRun,
    *,
    annotation_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the machine-readable report payload."""

    return {
        "dataset": {
            "name": run.dataset.dataset,
            "version": run.dataset.version,
            "source": run.dataset.source,
            "label_status": run.dataset.label_status,
        },
        "context": run.context,
        "metrics": {k: _round(v) for k, v in run.metrics.items()},
        "warnings": list(run.warnings),
        "annotation_stats": annotation_stats or {},
        "queries": [
            {
                "id": result.query_id,
                "text": result.text,
                "relevant_scene_ids": list(result.relevant_scene_ids),
                "ranked_scene_ids": list(result.ranked_scene_ids),
                "missing_relevant_scene_ids": list(result.missing_relevant_scene_ids),
                "metrics": {
                    "recall_at_5": _round(result.metrics.recall_at_5),
                    "recall_at_10": _round(result.metrics.recall_at_10),
                    "reciprocal_rank": _round(result.metrics.reciprocal_rank),
                    "ndcg_at_10": _round(result.metrics.ndcg_at_10),
                },
                "top_results": list(result.top_results),
            }
            for result in run.query_results
        ],
    }


def render_markdown(
    run: RetrievalRun,
    *,
    annotation_stats: dict[str, Any] | None = None,
) -> str:
    """Render a concise reviewer-facing Markdown report."""

    m = run.metrics
    lines = [
        "# Evaluation Report",
        "",
        f"Dataset: `{run.dataset.dataset}` v{run.dataset.version}",
        f"Config: `{run.context.get('config_path', '')}`",
        f"Queries: `{run.context.get('queries_path', '')}`",
        f"Index: `{run.context.get('embeddings_path', '')}`",
        f"Model: `{run.context.get('model', '')}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Query count | {m['query_count']} |",
        f"| Recall@5 | {m['recall_at_5']:.3f} |",
        f"| Recall@10 | {m['recall_at_10']:.3f} |",
        f"| MRR | {m['mrr']:.3f} |",
        f"| nDCG@10 | {m['ndcg_at_10']:.3f} |",
        "",
    ]

    if run.warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in run.warnings)
        lines.append("")

    if annotation_stats:
        lines.extend(
            [
                "## Annotation Correction Stats",
                "",
                "| Stat | Value |",
                "| --- | ---: |",
            ]
        )
        for key, value in annotation_stats.items():
            if isinstance(value, float):
                rendered = f"{value:.3f}"
            else:
                rendered = str(value)
            lines.append(f"| {key} | {rendered} |")
        lines.append("")

    lines.extend(["## Per-Query Results", ""])
    for result in run.query_results:
        top = ", ".join(result.ranked_scene_ids[:5])
        lines.extend(
            [
                f"### {result.query_id}: {result.text}",
                "",
                f"- Relevant: {', '.join(result.relevant_scene_ids)}",
                f"- Top scenes: {top or 'none'}",
                f"- Recall@5: {result.metrics.recall_at_5:.3f}",
                f"- MRR contribution: {result.metrics.reciprocal_rank:.3f}",
                f"- nDCG@10: {result.metrics.ndcg_at_10:.3f}",
                "",
            ]
        )
    return "\n".join(lines)


def write_reports(
    run: RetrievalRun,
    output_dir: str | Path,
    *,
    annotation_stats: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Write `summary.json` and `report.md`; return both paths."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "summary.json"
    md_path = out / "report.md"

    json_path.write_text(
        json.dumps(build_payload(run, annotation_stats=annotation_stats), indent=2),
        encoding="utf-8",
    )
    md_path.write_text(
        render_markdown(run, annotation_stats=annotation_stats),
        encoding="utf-8",
    )
    return json_path, md_path
