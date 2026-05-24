#!/usr/bin/env python3
"""Run retrieval evaluation against a configured Cinemateca index.

Supports three retriever modes (``--retriever {clip,bm25,hybrid}``) and a
batch sweep (``--all-modes``) that runs all three back-to-back and writes a
comparison table.

Films live under the per-film library layout
(``data/library/<slug>/{embeddings,metadata,frames}``); use ``--film-slug``
to point the harness at a specific film without authoring a per-film YAML.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CONFIG = REPO_ROOT / "config" / "default.yaml"
DEFAULT_QUERIES = REPO_ROOT / "data" / "eval" / "archive_demo_queries.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "eval" / "reports"
DEFAULT_FILM_SLUG = "edwin_porter-the_great_train_robbery_1903"


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate text retrieval over a Cinemateca index (CLIP / BM25 / hybrid)."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Config YAML to evaluate (default: config/default.yaml).",
    )
    parser.add_argument(
        "--queries",
        default=str(DEFAULT_QUERIES),
        help="Evaluation query YAML.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for per-mode JSON + comparison tables.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of ranked results to include in report output.",
    )
    parser.add_argument(
        "--film-slug",
        default=DEFAULT_FILM_SLUG,
        help=(
            "Per-film library slug whose embeddings/metadata to evaluate. "
            "Overrides cfg.paths.{embeddings,metadata,frames}_dir at runtime. "
            "Pass an empty string to evaluate the flat global layout."
        ),
    )
    parser.add_argument(
        "--retriever",
        choices=["clip", "bm25", "hybrid"],
        default="clip",
        help="Retriever to evaluate when not running --all-modes.",
    )
    parser.add_argument(
        "--all-modes",
        action="store_true",
        help="Evaluate clip / bm25 / hybrid in one run and emit a comparison table.",
    )
    parser.add_argument(
        "--sem-weight",
        type=float,
        default=0.5,
        help="Hybrid RRF: weight on the CLIP side (default 0.5).",
    )
    parser.add_argument(
        "--bm25-weight",
        type=float,
        default=0.5,
        help="Hybrid RRF: weight on the BM25 side (default 0.5).",
    )
    parser.add_argument(
        "--k-rrf",
        type=int,
        default=60,
        help="Hybrid RRF rank-shift constant (default 60).",
    )
    parser.add_argument(
        "--k-rrf-sweep",
        type=str,
        default="",
        help=(
            "Comma-separated k_rrf values for a sweep (e.g. '10,30,60,100'). "
            "Only honoured with --all-modes; emits a kRRF table in comparison.md."
        ),
    )
    return parser.parse_args(argv)


def _override_film_paths(cfg, slug: str) -> None:
    """Point cfg.paths.{embeddings,metadata,frames}_dir at one per-film dir.

    Mutates ``cfg.paths`` in place. The flat global layout is left alone
    when ``slug`` is empty.
    """
    if not slug:
        return
    library_dir = Path(getattr(cfg.paths, "library_dir", REPO_ROOT / "data" / "library"))
    film_dir = library_dir / slug
    if not film_dir.is_dir():
        raise SystemExit(
            f"Film slug not found under {library_dir}: {slug!r}. "
            f"Did you mean one of: {sorted(p.name for p in library_dir.iterdir() if p.is_dir())}?"
        )
    cfg.paths.embeddings_dir = str(film_dir / "embeddings")
    cfg.paths.metadata_dir = str(film_dir / "metadata")
    cfg.paths.frames_dir = str(film_dir / "frames")


def _serialize_run(run, payload_writer) -> dict[str, Any]:
    """Return a JSON-safe dict for a RetrievalRun (uses the existing builder)."""
    return payload_writer(run)


def _write_per_mode(run, out_path: Path, payload_writer) -> dict[str, Any]:
    payload = payload_writer(run)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _comparison_table(
    rows: list[dict[str, Any]],
    queries,
) -> str:
    """Render a per-query comparison table.

    ``rows`` is the list of mode-specific payloads (clip/bm25/hybrid).
    """
    header = (
        "| Query | CLIP R@10 | BM25 R@10 | Hybrid R@10 | "
        "CLIP nDCG@10 | BM25 nDCG@10 | Hybrid nDCG@10 |"
    )
    sep = "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    body: list[str] = []

    # Build lookup of query metrics per mode for fast retrieval.
    per_mode: dict[str, dict[str, dict[str, float]]] = {}
    for row in rows:
        mode_name = row["context"].get("retriever", "?")
        per_mode[mode_name] = {q["id"]: q["metrics"] for q in row["queries"]}

    for q in queries:
        qid = q.id
        cells = [f"`{qid}` {q.text[:48]}"]
        for metric_key in ("recall_at_10",):
            for mode in ("clip", "bm25", "hybrid"):
                m = per_mode.get(mode, {}).get(qid, {})
                cells.append(f"{m.get(metric_key, 0.0):.3f}")
        for metric_key in ("ndcg_at_10",):
            for mode in ("clip", "bm25", "hybrid"):
                m = per_mode.get(mode, {}).get(qid, {})
                cells.append(f"{m.get(metric_key, 0.0):.3f}")
        body.append("| " + " | ".join(cells) + " |")

    # Aggregate row: mean ± std across queries.
    import statistics

    agg_cells = ["**Mean ± std**"]
    for metric_key in ("recall_at_10",):
        for mode in ("clip", "bm25", "hybrid"):
            values = [per_mode.get(mode, {}).get(q.id, {}).get(metric_key, 0.0) for q in queries]
            mean = sum(values) / len(values) if values else 0.0
            std = statistics.pstdev(values) if values else 0.0
            agg_cells.append(f"{mean:.3f} ± {std:.3f}")
    for metric_key in ("ndcg_at_10",):
        for mode in ("clip", "bm25", "hybrid"):
            values = [per_mode.get(mode, {}).get(q.id, {}).get(metric_key, 0.0) for q in queries]
            mean = sum(values) / len(values) if values else 0.0
            std = statistics.pstdev(values) if values else 0.0
            agg_cells.append(f"{mean:.3f} ± {std:.3f}")
    body.append("| " + " | ".join(agg_cells) + " |")

    return "\n".join([header, sep, *body])


def _aggregate_table(rows: list[dict[str, Any]]) -> str:
    header = "| Retriever | Query count | Recall@5 | Recall@10 | MRR | nDCG@10 |"
    sep = "| --- | ---: | ---: | ---: | ---: | ---: |"
    body = []
    for row in rows:
        m = row["metrics"]
        name = row["context"].get("retriever", "?")
        body.append(
            "| {name} | {n} | {r5:.3f} | {r10:.3f} | {mrr:.3f} | {nd:.3f} |".format(
                name=name,
                n=m["query_count"],
                r5=m["recall_at_5"],
                r10=m["recall_at_10"],
                mrr=m["mrr"],
                nd=m["ndcg_at_10"],
            )
        )
    return "\n".join([header, sep, *body])


def _krrf_sweep_table(sweep_rows: list[dict[str, Any]]) -> str:
    header = "| k_rrf | sem_w | bm25_w | Recall@5 | Recall@10 | MRR | nDCG@10 |"
    sep = "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    body = []
    for r in sweep_rows:
        ctx = r["context"]
        m = r["metrics"]
        body.append(
            "| {k} | {sw:.2f} | {bw:.2f} | {r5:.3f} | {r10:.3f} | {mrr:.3f} | {nd:.3f} |".format(
                k=ctx.get("k_rrf", 0),
                sw=ctx.get("sem_w", 0.0),
                bw=ctx.get("bm25_w", 0.0),
                r5=m["recall_at_5"],
                r10=m["recall_at_10"],
                mrr=m["mrr"],
                nd=m["ndcg_at_10"],
            )
        )
    return "\n".join([header, sep, *body])


def _single_mode(args: argparse.Namespace) -> int:
    from cinemateca.config import load_config
    from cinemateca.eval.annotations import AnnotationStatsError, load_annotation_stats
    from cinemateca.eval.datasets import DatasetError, load_dataset
    from cinemateca.eval.report import write_reports
    from cinemateca.eval.retrieval import EvalError, run_retrieval_eval

    config_path = project_path(args.config)
    queries_path = project_path(args.queries)
    output_dir = project_path(args.output_dir)

    try:
        dataset = load_dataset(queries_path)
        cfg = load_config(config_path, project_root=REPO_ROOT)
        _override_film_paths(cfg, args.film_slug)
        run = run_retrieval_eval(
            cfg,
            dataset,
            config_path=config_path,
            top_k=args.top_k,
            retriever=args.retriever,
            sem_w=args.sem_weight,
            bm25_w=args.bm25_weight,
            k_rrf=args.k_rrf,
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

    print(f"Retriever: {args.retriever}")
    print(f"Queries: {len(dataset.queries)}")
    print(f"Recall@5:  {run.metrics['recall_at_5']:.3f}")
    print(f"Recall@10: {run.metrics['recall_at_10']:.3f}")
    print(f"MRR:       {run.metrics['mrr']:.3f}")
    print(f"nDCG@10:   {run.metrics['ndcg_at_10']:.3f}")
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    return 0


def _all_modes(args: argparse.Namespace) -> int:
    from cinemateca.config import load_config
    from cinemateca.eval.datasets import DatasetError, load_dataset
    from cinemateca.eval.report import build_payload
    from cinemateca.eval.retrieval import EvalError, run_retrieval_eval

    config_path = project_path(args.config)
    queries_path = project_path(args.queries)
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        dataset = load_dataset(queries_path)
        cfg = load_config(config_path, project_root=REPO_ROOT)
        _override_film_paths(cfg, args.film_slug)
    except (DatasetError, EvalError, FileNotFoundError) as exc:
        print(f"Evaluation setup failed: {exc}", file=sys.stderr)
        return 1

    mode_runs: list[dict[str, Any]] = []
    for mode in ("clip", "bm25", "hybrid"):
        try:
            run = run_retrieval_eval(
                cfg,
                dataset,
                config_path=config_path,
                top_k=args.top_k,
                retriever=mode,
                sem_w=args.sem_weight,
                bm25_w=args.bm25_weight,
                k_rrf=args.k_rrf,
            )
        except EvalError as exc:
            print(f"[{mode}] Evaluation failed: {exc}", file=sys.stderr)
            return 1
        payload = build_payload(run)
        out_path = output_dir / f"{mode}.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        mode_runs.append(payload)
        m = run.metrics
        print(
            f"[{mode}] R@5={m['recall_at_5']:.3f} R@10={m['recall_at_10']:.3f} "
            f"MRR={m['mrr']:.3f} nDCG@10={m['ndcg_at_10']:.3f}"
        )

    # Optional k_rrf sweep — only on hybrid, holding sem_w/bm25_w fixed.
    sweep_payloads: list[dict[str, Any]] = []
    if args.k_rrf_sweep:
        try:
            sweep_values = [int(x.strip()) for x in args.k_rrf_sweep.split(",") if x.strip()]
        except ValueError:
            print(f"Invalid --k-rrf-sweep: {args.k_rrf_sweep!r}", file=sys.stderr)
            return 1
        for k in sweep_values:
            try:
                run = run_retrieval_eval(
                    cfg,
                    dataset,
                    config_path=config_path,
                    top_k=args.top_k,
                    retriever="hybrid",
                    sem_w=args.sem_weight,
                    bm25_w=args.bm25_weight,
                    k_rrf=k,
                )
            except EvalError as exc:
                print(f"[hybrid k_rrf={k}] failed: {exc}", file=sys.stderr)
                return 1
            payload = build_payload(run)
            sweep_payloads.append(payload)
            m = run.metrics
            print(
                f"[hybrid k_rrf={k}] R@5={m['recall_at_5']:.3f} R@10={m['recall_at_10']:.3f} "
                f"MRR={m['mrr']:.3f} nDCG@10={m['ndcg_at_10']:.3f}"
            )
        sweep_path = output_dir / "krrf_sweep.json"
        sweep_path.write_text(json.dumps(sweep_payloads, indent=2), encoding="utf-8")

    # comparison.json — machine-readable aggregate of all runs.
    comparison_json = {
        "config": str(config_path),
        "queries": str(queries_path),
        "film_slug": args.film_slug,
        "top_k": args.top_k,
        "sem_weight": args.sem_weight,
        "bm25_weight": args.bm25_weight,
        "k_rrf": args.k_rrf,
        "modes": [
            {
                "retriever": payload["context"].get("retriever"),
                "metrics": payload["metrics"],
                "queries": [{"id": q["id"], "metrics": q["metrics"]} for q in payload["queries"]],
            }
            for payload in mode_runs
        ],
        "krrf_sweep": [
            {
                "k_rrf": payload["context"].get("k_rrf"),
                "sem_w": payload["context"].get("sem_w"),
                "bm25_w": payload["context"].get("bm25_w"),
                "metrics": payload["metrics"],
            }
            for payload in sweep_payloads
        ],
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison_json, indent=2), encoding="utf-8"
    )

    # comparison.md — human-readable.
    md_lines = [
        "# Retrieval ablation — CLIP vs BM25 vs Hybrid",
        "",
        f"Corpus: `{args.film_slug or 'flat'}`  |  Queries: `{queries_path.name}`  "
        f"|  top_k = {args.top_k}",
        "",
        "## Aggregate metrics",
        "",
        _aggregate_table(mode_runs),
        "",
        "## Per-query breakdown",
        "",
        _comparison_table(mode_runs, dataset.queries),
        "",
    ]
    if sweep_payloads:
        md_lines.extend(
            [
                "## k_rrf sweep (hybrid, " f"sem_w={args.sem_weight}, bm25_w={args.bm25_weight})",
                "",
                _krrf_sweep_table(sweep_payloads),
                "",
            ]
        )
    (output_dir / "comparison.md").write_text("\n".join(md_lines), encoding="utf-8")

    print()
    print(f"Wrote {output_dir / 'comparison.md'}")
    print(f"Wrote {output_dir / 'comparison.json'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.all_modes:
        return _all_modes(args)
    return _single_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())
