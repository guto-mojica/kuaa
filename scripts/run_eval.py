#!/usr/bin/env python3
"""Run retrieval evaluation against a configured Cinemateca index.

Supports three retriever modes (``--retriever {clip,bm25,hybrid}``) and a
batch sweep (``--all-modes``) that runs all three back-to-back and writes a
comparison table.

By default the harness evaluates the artifact paths from the selected config.
Use ``--film-slug`` to point it at a specific per-film library directory
(``data/library/<slug>/{embeddings,metadata,frames}``) without authoring a
per-film YAML.
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
DEFAULT_FILM_SLUG = ""


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
            "Default is an empty string, which evaluates the configured paths."
        ),
    )
    parser.add_argument(
        "--retriever",
        choices=["clip", "bm25", "hybrid"],
        default="clip",
        help="Retriever to evaluate when not running --all-modes.",
    )
    parser.add_argument(
        "--modality",
        choices=["text", "image", "rhyme", "all"],
        default="text",
        help=(
            "Eval modality. 'text' (default) runs the CLIP/BM25/hybrid retriever "
            "path unchanged. 'image|rhyme' score the matching slate "
            "(--queries must be an m3_full-shaped multimodal YAML). 'all' runs text "
            "plus every non-text modality, writing each to <output-dir>/<modality>.json."
        ),
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
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Global PRNG seed passed to seed_everything() and recorded in context "
            "for reproducibility. Default 0 (matches run_retrieval_eval default)."
        ),
    )
    return parser.parse_args(argv)


def _is_multimodal_yaml(queries_path: Path) -> bool:
    """True when the query YAML uses the m3_full multimodal shape.

    The discriminator is a ``query_type`` key on the first query entry — legacy
    text datasets (e.g. archive_demo_queries.yaml) never carry it, so they keep
    flowing through the strict ``load_dataset`` loader (the M3 regression pin),
    while an m3_full file is recognised and its text subset extracted below.
    """
    import yaml

    try:
        raw = yaml.safe_load(queries_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return False
    queries = raw.get("queries") if isinstance(raw, dict) else None
    if not isinstance(queries, list) or not queries:
        return False
    first = queries[0]
    return isinstance(first, dict) and "query_type" in first


def _text_dataset_from_multimodal(queries_path: Path):
    """Build a text-only EvaluationDataset from an m3_full multimodal YAML.

    Lets ``--modality text`` (which must keep using ``run_retrieval_eval``
    byte-for-byte) accept the same multimodal file the other modalities take:
    only the ``query_type == "text"`` entries — which carry the maintainer's
    ``relevant_scene_ids`` / ``relevance`` hypotheses — become ``QueryCase``
    rows. Non-text entries are ignored. Raises if there are no text queries.
    """
    from cinemateca.eval.datasets import DatasetError, EvaluationDataset, QueryCase
    from cinemateca.eval.slates import load_modal_queries
    from cinemateca.scene_ids import scene_id_key

    modal = load_modal_queries(queries_path, only_types={"text"})
    cases = []
    for q in modal:
        if q.query_type != "text" or not q.text:
            continue
        relevant = tuple(scene_id_key(s) for s in q.relevant_scene_ids)
        if not relevant:
            continue
        relevance = {scene_id_key(k): float(v) for k, v in q.relevance.items()} or {
            sid: 1.0 for sid in relevant
        }
        cases.append(
            QueryCase(
                id=q.id,
                text=q.text,
                relevant_scene_ids=relevant,
                relevance=relevance,
                notes=q.notes or "",
            )
        )
    if not cases:
        raise DatasetError(f"no text queries with relevant_scene_ids in {queries_path}")
    return EvaluationDataset(
        dataset="m3_text",
        version=1,
        queries=tuple(cases),
        source={"modality": "text", "from": str(queries_path)},
        label_status="seed_curator_grading_pending",
        path=queries_path,
    )


def _load_text_dataset(queries_path: Path):
    """Load a text EvaluationDataset, auto-detecting legacy vs m3_full shape."""
    if _is_multimodal_yaml(queries_path):
        return _text_dataset_from_multimodal(queries_path)
    from cinemateca.eval.datasets import load_dataset

    return load_dataset(queries_path)


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
    from cinemateca.eval.datasets import DatasetError
    from cinemateca.eval.report import write_reports
    from cinemateca.eval.retrieval import EvalError, run_retrieval_eval

    config_path = project_path(args.config)
    queries_path = project_path(args.queries)
    output_dir = project_path(args.output_dir)

    try:
        dataset = _load_text_dataset(queries_path)
        cfg = load_config(config_path, project_root=REPO_ROOT)
        from cinemateca.reproducibility import seed_everything

        seed_everything(cfg.seed)
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
            seed=args.seed,
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
    from cinemateca.eval.datasets import DatasetError
    from cinemateca.eval.report import build_payload
    from cinemateca.eval.retrieval import EvalError, run_retrieval_eval

    config_path = project_path(args.config)
    queries_path = project_path(args.queries)
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Use the auto-detecting loader (legacy vs m3_full multimodal), same as
        # the text path below — the strict load_dataset() rejects m3_full image
        # rows that legitimately carry no relevant_scene_ids (review #2).
        dataset = _load_text_dataset(queries_path)
        cfg = load_config(config_path, project_root=REPO_ROOT)
        from cinemateca.reproducibility import seed_everything

        seed_everything(cfg.seed)
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
                seed=args.seed,
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
                    seed=args.seed,
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
        "seed": args.seed,
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


# ─────────────────────────────────────────────────────────────────────────────
# Non-text modality routing (E3b): image / rhyme.
#
# These dispatch to the per-modality scorers in cinemateca.eval.retrieval, which
# call the REAL retrieval backend (CLIP find / find_rhymes) via
# cinemateca.eval.slates.generate_slate and return the same RetrievalRun the text
# path produces — so the existing report writer serialises them unchanged. The
# text path (_single_mode / _all_modes) is left untouched.
# ─────────────────────────────────────────────────────────────────────────────

_MODAL_SCORERS = ("image", "rhyme")


def _run_one_modality(cfg, queries, modality: str, args: argparse.Namespace, config_path: Path):
    """Run one non-text modality scorer and return its RetrievalRun."""
    from cinemateca.eval import retrieval as _retr

    scorers = {
        "image": _retr.run_image_eval,
        "rhyme": _retr.run_rhyme_eval,
    }
    library_dir = Path(getattr(cfg.paths, "library_dir", REPO_ROOT / "data" / "library"))
    run = scorers[modality](
        cfg,
        queries,
        library_dir=library_dir,
        film_slug=args.film_slug or None,
        seed=args.seed,
        top_k=args.top_k,
    )
    # Record the actual config + queries paths the text path also reports.
    run.context["config_path"] = str(config_path)
    run.context["queries_path"] = str(project_path(args.queries))
    return run


def _modal_mode(args: argparse.Namespace) -> int:
    """Score a single non-text modality; write summary.json + report.md."""
    from cinemateca.config import load_config
    from cinemateca.errors import EvalError
    from cinemateca.eval.report import write_reports
    from cinemateca.eval.slates import load_modal_queries

    config_path = project_path(args.config)
    queries_path = project_path(args.queries)
    output_dir = project_path(args.output_dir)

    try:
        cfg = load_config(config_path, project_root=REPO_ROOT)
        from cinemateca.reproducibility import seed_everything

        seed_everything(cfg.seed)
        queries = load_modal_queries(queries_path)
        run = _run_one_modality(cfg, queries, args.modality, args, config_path)
        json_path, md_path = write_reports(run, output_dir)
    except (EvalError, FileNotFoundError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1

    m = run.metrics
    print(f"Modality: {args.modality}")
    print(f"Queries scored: {m['query_count']}")
    print(f"Relevance method: {run.context.get('relevance_method', '?')}")
    print(f"Recall@5:  {m['recall_at_5']:.3f}")
    print(f"Recall@10: {m['recall_at_10']:.3f}")
    print(f"MRR:       {m['mrr']:.3f}")
    print(f"nDCG@10:   {m['ndcg_at_10']:.3f}")
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    return 0


def _all_modalities(args: argparse.Namespace) -> int:
    """Run text (via --retriever) + every non-text modality.

    The text run writes <output-dir>/text.json; each non-text modality writes
    <output-dir>/<modality>.json (mirroring _all_modes's {mode}.json convention).
    A non-text modality that cannot be scored (e.g. its slate is empty) is
    reported as a warning and skipped rather than aborting the whole sweep.
    """
    from cinemateca.config import load_config
    from cinemateca.eval.datasets import DatasetError
    from cinemateca.eval.report import build_payload
    from cinemateca.eval.retrieval import EvalError, run_retrieval_eval
    from cinemateca.eval.slates import load_modal_queries

    config_path = project_path(args.config)
    queries_path = project_path(args.queries)
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        cfg = load_config(config_path, project_root=REPO_ROOT)
        from cinemateca.reproducibility import seed_everything

        seed_everything(cfg.seed)
    except (EvalError, FileNotFoundError) as exc:
        print(f"Evaluation setup failed: {exc}", file=sys.stderr)
        return 1

    # Text run (regression-preserving: same dataset loader + film-path override).
    try:
        dataset = _load_text_dataset(queries_path)
        text_cfg = load_config(config_path, project_root=REPO_ROOT)
        _override_film_paths(text_cfg, args.film_slug)
        text_run = run_retrieval_eval(
            text_cfg,
            dataset,
            config_path=config_path,
            top_k=args.top_k,
            retriever=args.retriever,
            sem_w=args.sem_weight,
            bm25_w=args.bm25_weight,
            k_rrf=args.k_rrf,
            seed=args.seed,
        )
        (output_dir / "text.json").write_text(
            json.dumps(build_payload(text_run), indent=2), encoding="utf-8"
        )
        m = text_run.metrics
        print(
            f"[text/{args.retriever}] R@5={m['recall_at_5']:.3f} R@10={m['recall_at_10']:.3f} "
            f"MRR={m['mrr']:.3f} nDCG@10={m['ndcg_at_10']:.3f}"
        )
    except (DatasetError, EvalError, FileNotFoundError) as exc:
        print(f"[text] Evaluation failed: {exc}", file=sys.stderr)
        return 1

    # Non-text modalities — each may legitimately have an empty slate on a
    # given corpus (e.g. a rhyme anchor with no cross-film neighbours); warn + continue.
    try:
        modal_queries = load_modal_queries(queries_path)
    except EvalError as exc:
        print(f"[modalities] could not load modal queries: {exc}", file=sys.stderr)
        return 1

    for modality in _MODAL_SCORERS:
        try:
            run = _run_one_modality(cfg, modal_queries, modality, args, config_path)
        except EvalError as exc:
            print(f"[{modality}] skipped: {exc}", file=sys.stderr)
            continue
        (output_dir / f"{modality}.json").write_text(
            json.dumps(build_payload(run), indent=2), encoding="utf-8"
        )
        mm = run.metrics
        print(
            f"[{modality}] n={mm['query_count']} R@5={mm['recall_at_5']:.3f} "
            f"R@10={mm['recall_at_10']:.3f} MRR={mm['mrr']:.3f} nDCG@10={mm['ndcg_at_10']:.3f} "
            f"({run.context.get('relevance_method', '?')})"
        )

    print()
    print(f"Wrote per-modality reports to {output_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.modality == "all":
        return _all_modalities(args)
    if args.modality in _MODAL_SCORERS:
        return _modal_mode(args)
    # modality == "text": preserve the existing retriever path byte-for-byte.
    if args.all_modes:
        return _all_modes(args)
    return _single_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())
