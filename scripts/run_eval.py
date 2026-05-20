#!/usr/bin/env python3
"""Run M2 retrieval evaluation against a configured Cinemateca index."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CONFIG = REPO_ROOT / "config" / "demo.yaml"
DEFAULT_QUERIES = REPO_ROOT / "data" / "eval" / "archive_demo_queries.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "eval" / "reports"


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate text retrieval over a Cinemateca CLIP index."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Config YAML to evaluate, usually config/demo.yaml.",
    )
    parser.add_argument(
        "--queries",
        default=str(DEFAULT_QUERIES),
        help="Evaluation query YAML.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for summary.json and report.md.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of ranked results to include in report output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = project_path(args.config)
    queries_path = project_path(args.queries)
    output_dir = project_path(args.output_dir)

    from cinemateca.config import load_config
    from cinemateca.eval.annotations import AnnotationStatsError, load_annotation_stats
    from cinemateca.eval.datasets import DatasetError, load_dataset
    from cinemateca.eval.report import write_reports
    from cinemateca.eval.retrieval import EvalError, run_retrieval_eval

    try:
        dataset = load_dataset(queries_path)
        cfg = load_config(config_path, project_root=REPO_ROOT)
        run = run_retrieval_eval(
            cfg,
            dataset,
            config_path=config_path,
            top_k=args.top_k,
        )
        annotation_stats = load_annotation_stats(cfg.paths.metadata_dir).to_dict()
        json_path, md_path = write_reports(
            run,
            output_dir,
            annotation_stats=annotation_stats,
        )
    except (AnnotationStatsError, DatasetError, EvalError, FileNotFoundError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1

    print("Evaluation complete")
    print(f"Queries: {len(dataset.queries)}")
    print(f"Recall@5: {run.metrics['recall_at_5']:.3f}")
    print(f"Recall@10: {run.metrics['recall_at_10']:.3f}")
    print(f"MRR: {run.metrics['mrr']:.3f}")
    print(f"nDCG@10: {run.metrics['ndcg_at_10']:.3f}")
    print(f"Annotated scenes: {annotation_stats['scenes_with_manual_annotations']}")
    print(f"Correction rate: {annotation_stats['correction_rate']:.3f}")
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
