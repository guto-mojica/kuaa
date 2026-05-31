#!/usr/bin/env python3
"""Surface the worst-scoring text queries from the eval as a failure-analysis doc.

Runs the three text retrievers (CLIP / BM25 / hybrid) over the text subset of a
multimodal eval YAML against one film's live index, groups each query's results
across retrievers into a per-query record (enriching the top *non-relevant*
results with the REAL Moondream caption the retriever saw, read from on-disk
metadata), picks the ``--n`` worst by nDCG@10, and writes the rendered stubs into
a delimited M4 block of ``--out`` (default ``docs/FAILURE_ANALYSIS.md``),
preserving any existing content (the M2 cases).

The emitted stubs have empty Hypothesis / Mitigation lines — a human fills those
from the real ranks + captions the tool surfaced (proxy-labelled: the relevant
scene ids are the maintainer's pre-curator HYPOTHESES from the YAML, not curator
grades — the M4 preamble says so).

Example (live SigLIP2 index, GPU):

    uv run python scripts/analyze_failures.py \\
        --queries data/eval/m3_full_queries.yaml \\
        --library-dir data/library --seed 0 --n 8 \\
        --out docs/FAILURE_ANALYSIS.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_QUERIES = REPO_ROOT / "data" / "eval" / "m3_full_queries.yaml"
DEFAULT_LIBRARY_DIR = REPO_ROOT / "data" / "library"
DEFAULT_OUT = REPO_ROOT / "docs" / "FAILURE_ANALYSIS.md"
DEFAULT_FILM_SLUG = "jeca_tatu"
RETRIEVERS = ("clip", "bm25", "hybrid")

# Markers delimiting the auto-generated M4 block. Re-running the tool replaces
# only the text between them; everything else in the doc (the M2 cases) is left
# byte-for-byte untouched.
M4_BEGIN = "<!-- BEGIN M4 AUTO-FAILURE-CASES -->"
M4_END = "<!-- END M4 AUTO-FAILURE-CASES -->"


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--queries", default=str(DEFAULT_QUERIES), help="Multimodal eval YAML.")
    parser.add_argument(
        "--library-dir",
        default=str(DEFAULT_LIBRARY_DIR),
        help="Per-film library root (data/library). The film selected by --film-slug is evaluated.",
    )
    parser.add_argument(
        "--film-slug",
        default=DEFAULT_FILM_SLUG,
        help=(
            "Film whose live index the text retrievers run against "
            f"(default: {DEFAULT_FILM_SLUG}). run_retrieval_eval scores one index."
        ),
    )
    parser.add_argument("--config", default=str(REPO_ROOT / "config" / "default.yaml"))
    parser.add_argument(
        "--seed", type=int, default=0, help="PRNG seed (recorded for reproducibility)."
    )
    parser.add_argument("--n", type=int, default=8, help="Number of worst queries to surface.")
    parser.add_argument("--top-k", type=int, default=10, help="Ranked results per retriever.")
    parser.add_argument(
        "--out", default=str(DEFAULT_OUT), help="Markdown doc to write the M4 block into."
    )
    return parser.parse_args(argv)


def _first_relevant_rank(ranked: tuple[str, ...], relevant: set[str]) -> int | None:
    """1-based rank of the first relevant scene in ``ranked``, or None."""
    for idx, sid in enumerate(ranked, start=1):
        if sid in relevant:
            return idx
    return None


def _build_records(
    runs_by_retriever: dict[str, Any],
    desc_by_scene: dict[str, Any],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    """Group per-query RetrievalResults across retrievers into failure records.

    The hybrid run is the reference for query text, the ranked metric, the
    top-wrong list and the missing-relevant set (it is the production default
    retriever); ``first_relevant_rank_by_retriever`` spans all three. The
    top-wrong rows are the hybrid's top-K ranked scenes that are NOT relevant,
    each enriched with the on-disk Moondream caption (the text path's
    ``top_results`` carry no ``description`` key — see retrieval._result_rows).
    """
    by_qid: dict[str, dict[str, Any]] = {}
    reference = "hybrid" if "hybrid" in runs_by_retriever else RETRIEVERS[0]

    # Index every retriever's per-query result by query id.
    per_retriever: dict[str, dict[str, Any]] = {}
    for name, run in runs_by_retriever.items():
        per_retriever[name] = {r.query_id: r for r in run.query_results}

    ref_results = per_retriever[reference]
    for qid, ref in ref_results.items():
        relevant = set(ref.relevant_scene_ids)
        ranks: dict[str, int | None] = {}
        ever_retrieved: set[str] = set()
        for name in RETRIEVERS:
            res = per_retriever.get(name, {}).get(qid)
            if res is None:
                ranks[name] = None
                continue
            ranks[name] = _first_relevant_rank(res.ranked_scene_ids, relevant)
            ever_retrieved.update(res.ranked_scene_ids)

        # Top-wrong = reference's top-K ranked scenes NOT in the relevant set,
        # enriched with the real caption from metadata.
        top_wrong: list[dict[str, Any]] = []
        for row in ref.top_results[:top_k]:
            sid = str(row.get("scene_id", ""))
            if sid in relevant:
                continue
            desc_entry = desc_by_scene.get(sid) or {}
            description = (
                str(desc_entry.get("description", "")) if isinstance(desc_entry, dict) else ""
            )
            top_wrong.append({"scene_id": sid, "description": description})

        # Missing relevant = relevant ids that NO retriever surfaced in top-K.
        missing = tuple(sid for sid in ref.relevant_scene_ids if sid not in ever_retrieved)

        by_qid[qid] = {
            "query_id": qid,
            "query_text": ref.text,
            "metrics": {
                "ndcg_at_10": ref.metrics.ndcg_at_10,
                "recall_at_5": ref.metrics.recall_at_5,
                "recall_at_10": ref.metrics.recall_at_10,
                "reciprocal_rank": ref.metrics.reciprocal_rank,
            },
            "first_relevant_rank_by_retriever": ranks,
            "top_wrong": tuple(top_wrong),
            "missing_relevant": missing,
        }
    return list(by_qid.values())


def _render_m4_block(cases, *, context: dict[str, Any]) -> str:
    """Render the M4 section body (preamble + one stub per FailureCase)."""
    from cinemateca.eval.failures import FailureCase  # noqa: F401  (type only)

    preamble = (
        f"## M4 — Multi-modal failure cases ({context['film_slug']}, live "
        f"{context['model']})\n\n"
        "> **Proxy-labelled, not curator-graded.** The relevant-scene ids below are "
        "the maintainer's pre-curator HYPOTHESES from `data/eval/m3_full_queries.yaml`, "
        "not human grades. Ranks and Moondream captions are real (the live "
        f"{context['model']} index over the `{context['film_slug']}` film, "
        f"seed={context['seed']}, top_k={context['top_k']}). Curator grades (WS-4 E5) "
        "will refine which scenes count as relevant; the failure *patterns* surfaced "
        "here are robust to that.\n\n"
        "The three text retrievers (CLIP / BM25 / hybrid) are scored on the "
        f"{context['query_count']} HY-labelled text queries; the "
        f"{context['n']} worst by nDCG@10 are below. Ranks are 1-based; \"—\" means "
        "the first relevant scene never appeared in that retriever's top-"
        f"{context['top_k']}. Each top non-relevant result cites the exact Moondream "
        "caption the retriever ranked.\n"
    )
    parts = [preamble]
    for case in cases:
        parts.append(case.to_markdown_stub())
    return "\n".join(parts).rstrip() + "\n"


def _splice_into_doc(out_path: Path, m4_block: str) -> None:
    """Write/replace the delimited M4 block in ``out_path``, preserving the rest.

    If the markers already exist, only the text between them is replaced. Else
    the block is appended after the existing content (the M2 cases) under a rule.
    """
    new_section = f"{M4_BEGIN}\n{m4_block}{M4_END}\n"
    if not out_path.exists():
        out_path.write_text(new_section, encoding="utf-8")
        return

    existing = out_path.read_text(encoding="utf-8")
    if M4_BEGIN in existing and M4_END in existing:
        head, _, rest = existing.partition(M4_BEGIN)
        _, _, tail = rest.partition(M4_END)
        # Drop a single leading newline on the tail to avoid blank-line creep.
        tail = tail[1:] if tail.startswith("\n") else tail
        out_path.write_text(head + new_section + tail, encoding="utf-8")
    else:
        glue = "" if existing.endswith("\n\n") else ("\n" if existing.endswith("\n") else "\n\n")
        out_path.write_text(existing + glue + "---\n\n" + new_section, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    from cinemateca.config import load_config
    from cinemateca.errors import EvalError
    from cinemateca.eval.failures import worst_queries
    from cinemateca.library import load_metadata
    from cinemateca.reproducibility import seed_everything

    # Reuse run_eval's text-subset extraction (the E3b loader) verbatim.
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from run_eval import _text_dataset_from_multimodal  # type: ignore[import-not-found]

    args = parse_args(argv)
    queries_path = project_path(args.queries)
    library_dir = project_path(args.library_dir)
    out_path = project_path(args.out)
    film_dir = library_dir / args.film_slug
    if not film_dir.is_dir():
        print(f"Film slug not found under {library_dir}: {args.film_slug!r}", file=sys.stderr)
        return 1

    try:
        dataset = _text_dataset_from_multimodal(queries_path)
        cfg = load_config(project_path(args.config), project_root=REPO_ROOT)
        seed_everything(args.seed)
        # Point the text retrievers at THIS film's live index (SigLIP2 by config).
        cfg.paths.embeddings_dir = str(film_dir / "embeddings")
        cfg.paths.metadata_dir = str(film_dir / "metadata")
        cfg.paths.frames_dir = str(film_dir / "frames")

        from cinemateca.eval.retrieval import run_retrieval_eval

        runs: dict[str, Any] = {}
        for retr in RETRIEVERS:
            runs[retr] = run_retrieval_eval(
                cfg,
                dataset,
                config_path=None,
                top_k=args.top_k,
                retriever=retr,
                seed=args.seed,
            )
        # Real Moondream captions for the wrong-result enrichment.
        _kf, desc_by_scene, _vis, _tags = load_metadata(Path(cfg.paths.metadata_dir))
    except (EvalError, FileNotFoundError) as exc:
        print(f"Failure analysis failed: {exc}", file=sys.stderr)
        return 1

    records = _build_records(runs, desc_by_scene, top_k=args.top_k)
    cases = worst_queries(records, n=args.n, by="ndcg_at_10")

    context = {
        "film_slug": args.film_slug,
        "model": runs["clip"].context.get("model", "?"),
        "seed": args.seed,
        "top_k": args.top_k,
        "n": args.n,
        "query_count": len(dataset.queries),
    }
    m4_block = _render_m4_block(cases, context=context)
    _splice_into_doc(out_path, m4_block)

    print(f"Film: {args.film_slug}  Model: {context['model']}")
    print(f"Text queries scored: {len(dataset.queries)}  |  worst surfaced: {len(cases)}")
    for c in cases:
        ranks = c.first_relevant_rank_by_retriever
        print(
            f"  {c.query_id:<9} nDCG@10={c.metric_value:.3f}  "
            f"ranks(clip/bm25/hybrid)="
            f"{ranks.get('clip')}/{ranks.get('bm25')}/{ranks.get('hybrid')}  "
            f'"{c.query_text[:48]}"'
        )
    print(f"Wrote M4 block → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
