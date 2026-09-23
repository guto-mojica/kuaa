#!/usr/bin/env python3
"""Generate the proxy-first retriever-variant ablation table.

Runs :func:`kuaa.eval.ablation.run_ablation` over the 15 text queries in
``data/eval/m3_full_queries.yaml`` (the common HY-labelled set) and writes the
result into ``docs/EVALUATION_RESULTS.md``. The generated block is delimited by
HTML comment markers (``<!-- ABLATION START -->`` / ``<!-- ABLATION END -->``)
so re-runs replace only that block — any other content in the doc (above or
below the markers) is left untouched.

Every published cell is a REAL proxy number computed on the demo corpus, or a
literal ``pending (...)`` for a row whose backend is not wired. Under
``--no-rerank`` the cross-encoder row is ``pending (rerank off)``;
``--with-rerank`` fills it by running the production
``find(mode="hybrid", rerank=True)`` path.

Usage::

    uv run python scripts/run_ablation.py \
        --queries data/eval/m3_full_queries.yaml \
        --library-dir data/library --seed 0 --no-rerank \
        --out docs/EVALUATION_RESULTS.md
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_QUERIES = REPO_ROOT / "data" / "eval" / "m3_full_queries.yaml"
DEFAULT_LIBRARY = REPO_ROOT / "data" / "library"
DEFAULT_OUT = REPO_ROOT / "docs" / "EVALUATION_RESULTS.md"
DEFAULT_CONFIG = REPO_ROOT / "config" / "default.yaml"

_ABLATION_START = "<!-- ABLATION START -->"
_ABLATION_END = "<!-- ABLATION END -->"


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", default=str(DEFAULT_QUERIES), help="Multimodal query YAML.")
    parser.add_argument(
        "--library-dir", default=str(DEFAULT_LIBRARY), help="Per-film library root."
    )
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG), help="Config YAML (default: config/default.yaml)."
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help="Markdown doc whose ablation section to write/replace (other content untouched).",
    )
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed (default 0).")
    rerank = parser.add_mutually_exclusive_group()
    rerank.add_argument(
        "--with-rerank",
        dest="with_rerank",
        action="store_true",
        help="Compute the hybrid_rerank row via the production find() path.",
    )
    rerank.add_argument(
        "--no-rerank",
        dest="with_rerank",
        action="store_false",
        help="Render the hybrid_rerank row as pending — the default.",
    )
    parser.set_defaults(with_rerank=False)
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print the ablation markdown to stdout without touching the doc.",
    )
    parser.add_argument(
        "--grades",
        default=None,
        metavar="RUN_ID_OR_PATH",
        help=(
            "Use human grades from this grading run instead of proxy labels. "
            "Accepts a run ID (looked up in data/eval/) or an absolute path to "
            "a run JSONL file. When provided, queries present in the grade log "
            "use human-validated relevance; absent queries fall back to proxy. "
            "Without --grades, behavior is byte-for-byte unchanged (proxy only)."
        ),
    )
    return parser.parse_args(argv)


def _build_ablation_section(
    table_md: str,
    *,
    with_rerank: bool,
    seed: int,
    queries: Path,
    grades_arg: str | None = None,
    query_count: int | None = None,
) -> str:
    """Wrap the rendered ablation table in the delimited doc section.

    ``grades_arg`` switches the surrounding prose to the human-graded reading.
    The section is published verbatim, so proxy caveats around a graded table
    (or the reverse) would misdescribe the numbers a reader is looking at.
    """
    mode = "with-rerank (cross-encoder live)" if with_rerank else "no-rerank (rerank row pending)"
    set_size = f"{query_count} text queries" if query_count else "the text queries"
    if grades_arg:
        return _graded_section(
            table_md,
            mode=mode,
            seed=seed,
            queries=queries,
            grades_arg=grades_arg,
            set_size=set_size,
            with_rerank=with_rerank,
        )
    lines = [
        _ABLATION_START,
        "",
        "## Retriever-variant proxy ablation (SigLIP2 default)",
        "",
        f"**Run date:** {date.today().isoformat()} — `scripts/run_ablation.py` "
        f"({mode}, seed={seed}).",
        f"**Query set:** `{queries.name}` — {set_size} (common set).",
        "",
        "Retriever-variant ablation on a **common query set with the same proxy "
        "labels** (apples-to-apples). This is the launch ablation: it is producible "
        "with **zero human grades** and every cell is either a real proxy number or "
        "an honest `pending (...)`.",
        "",
        table_md,
        "",
        "**Reading the numbers** (proxy / HY, not human-graded):",
        "",
        "- **Treat differences here as noise unless they are large.** This is 15 "
        "queries scored against the maintainer's pre-curator *hypothesis* labels, "
        "with roughly three relevant scenes each out of ~450. Every retriever "
        "lands near nDCG@10 ≈ 0.08, where one query's movement swings the third "
        "decimal. The table is a wiring check — it tells you a leg is connected "
        "and roughly not harmful — not a quality ranking.",
        "",
        "- **It cannot see the PT/EN gap at all.** 8 of the 15 text queries are "
        "Portuguese, and the labels were authored against what the system used to "
        "return. Before the bilingual index expansion those PT queries retrieved "
        "*nothing* from BM25; they now retrieve plausible scenes, which these "
        "labels neither reward nor penalise. Use `scripts/check_pt_parity.py` for "
        "that axis.",
        "",
        "- **`hybrid` vs `hybrid_no_metadata` is finally a real comparison.** The two "
        "rows were byte-identical for as long as they shipped, because the "
        "metadata scorer bailed out on any query over 4 tokens and so returned `{}` "
        "on 13 of these 15 queries — the ablation was subtracting a leg that was "
        "already absent.",
        "",
        "- **Nothing here measures short object queries**, which is the case the "
        "metadata leg exists for. Its fusion share now tapers with query length "
        "(`kuaa.retrieval.hybrid.effective_metadata_w`); this slate only exercises "
        "the long end of that taper.",
        "",
        "Reproduce:",
        "",
        "```bash",
        "uv run python scripts/run_ablation.py \\",
        f"  --queries {_rel(queries)} --library-dir data/library \\",
        f"  --seed {seed} --{'with' if with_rerank else 'no'}-rerank \\",
        "  --out docs/EVALUATION_RESULTS.md",
        "```",
        "",
        _ABLATION_END,
    ]
    return "\n".join(lines)


def _graded_section(
    table_md: str,
    *,
    mode: str,
    seed: int,
    queries: Path,
    grades_arg: str,
    set_size: str,
    with_rerank: bool,
) -> str:
    """The doc section for a table scored on human grades."""
    lines = [
        _ABLATION_START,
        "",
        "## Retriever-variant ablation, human-graded (SigLIP2 default)",
        "",
        f"**Run date:** {date.today().isoformat()} — `scripts/run_ablation.py` "
        f"({mode}, seed={seed}, `--grades {grades_arg}`).",
        f"**Query set:** `{queries.name}` — {set_size}, each scored against the "
        f"grades recorded for it in run `{grades_arg}`.",
        "",
        "Every row retrieves through the library-wide path the app serves "
        "(`kuaa.search.aggregate`), which is also the path the graded pool was "
        "drawn from — so each row is scored on candidates a human actually "
        "judged rather than on the pool's leftovers.",
        "",
        table_md,
        "",
        "**Reading the numbers:**",
        "",
        "- **Relevance is pooled, so recall is recall over judged scenes.** A "
        "candidate no retriever in the pool proposed was never shown to a grader "
        "and counts as irrelevant. That is the standard pooled-evaluation "
        "contract; it means these numbers compare the pooled variants to each "
        "other, and do not estimate absolute recall over the corpus.",
        "",
        "- **`hybrid` vs `hybrid_no_metadata` isolates the exact-lexical leg.** "
        "The two rows differ only in `metadata_w`, so the delta is that signal's "
        "contribution and nothing else.",
        "",
        "- **The rerank row sits on the `hybrid` row's candidates.** It widens "
        "the first stage to the reranker's input window, reorders, and cuts back "
        "— compare it to `hybrid`, not to `clip`.",
        "",
        "- **PT/EN is a design axis of the query set, not of this table.** The "
        "pt-NN / en-NN pairs make the gap measurable per query; "
        "`scripts/check_pt_parity.py` is the surface that reports it.",
        "",
        "Reproduce:",
        "",
        "```bash",
        "uv run python scripts/run_ablation.py \\",
        f"  --queries {_rel(queries)} --library-dir data/library \\",
        f"  --seed {seed} --{'with' if with_rerank else 'no'}-rerank --grades {grades_arg} \\",
        "  --out docs/EVALUATION_RESULTS.md",
        "```",
        "",
        _ABLATION_END,
    ]
    return "\n".join(lines)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _merge_into_doc(doc_path: Path, ablation_section: str) -> None:
    """Write the ablation section into ``doc_path``, preserving everything else.

    If the doc already has the ablation markers, the block between them is replaced.
    Otherwise the section is appended (with a leading blank-line separator).
    Content outside the markers (before or after) is never modified.
    """
    if doc_path.exists():
        existing = doc_path.read_text(encoding="utf-8")
    else:
        existing = ""

    if _ABLATION_START in existing and _ABLATION_END in existing:
        head, _, rest = existing.partition(_ABLATION_START)
        _, _, tail = rest.partition(_ABLATION_END)
        merged = head.rstrip() + "\n\n" + ablation_section + tail.rstrip() + "\n"
    else:
        sep = "\n\n---\n\n" if existing.strip() else ""
        merged = existing.rstrip() + sep + ablation_section + "\n"

    doc_path.write_text(merged, encoding="utf-8")


def _load_graded_labels(
    grades_arg: str | None,
) -> tuple[dict | None, str | None]:
    """Load graded labels from a run ID or path; return (graded_labels, validated_label).

    ``graded_labels`` is ``{query_id: {scene_id: float_grade}}`` carrying every
    grade, SKIP (-1) included: the ablation needs the SKIPs to exclude those
    scenes from the ranking rather than score them as irrelevant, and it
    decides itself what a query with no positive grade does. Returns
    ``(None, None)`` when no ``--grades`` arg.
    """
    if grades_arg is None:
        return None, None

    from pathlib import Path as _Path

    from kuaa.eval.grades import EvalRun, export_run

    path = _Path(grades_arg)
    if path.is_absolute() and path.exists():
        run = EvalRun(run_id=path.stem, root=path.parent)
    else:
        # Treat as run_id under data/eval/.
        run_root = REPO_ROOT / "data" / "eval"
        run = EvalRun(run_id=grades_arg, root=run_root)

    exported = export_run(run)
    # graded_labels: {query_id: {scene_id: float_grade}}
    graded_labels: dict = {
        qid: {sid: float(g) for sid, g in scenes.items()}
        for qid, scenes in exported["grades"].items()
    }

    distinct = exported["summary"]["distinct_pairs"]
    validated_label = f"human-validated (run {run.run_id}, n={distinct} grades)"
    return graded_labels, validated_label


def _scored_query_count(table) -> int | None:
    """Queries actually scored, read off the first computed row."""
    for _cfg, metrics in table.rows:
        if metrics:
            count = metrics.get("query_count")
            if count:
                return int(count)
    return None


def main(argv: list[str] | None = None) -> int:
    from kuaa.config import load_config
    from kuaa.errors import EvalError
    from kuaa.eval.ablation import (
        DEFAULT_ABLATION_CONFIGS,
        DEFAULT_ABLATION_CONFIGS_NO_RERANK,
        run_ablation,
    )
    from kuaa.eval.slates import load_modal_queries

    args = parse_args(argv)
    queries_path = project_path(args.queries)
    library_dir = project_path(args.library_dir)
    config_path = project_path(args.config)
    out_path = project_path(args.out)

    graded_labels, validated_label = _load_graded_labels(getattr(args, "grades", None))

    try:
        cfg = load_config(config_path, project_root=REPO_ROOT, ensure_dirs=False)
        queries = load_modal_queries(queries_path)
        configs = (
            DEFAULT_ABLATION_CONFIGS if args.with_rerank else DEFAULT_ABLATION_CONFIGS_NO_RERANK
        )
        table = run_ablation(
            cfg,
            library_dir=library_dir,
            queries=queries,
            configs=configs,
            seed=args.seed,
            graded_labels=graded_labels,
            validated_label=validated_label,
        )
    except (EvalError, FileNotFoundError) as exc:
        print(f"Ablation failed: {exc}", file=sys.stderr)
        return 1

    table_md = table.to_markdown()
    ablation_section = _build_ablation_section(
        table_md,
        with_rerank=args.with_rerank,
        seed=args.seed,
        queries=queries_path,
        grades_arg=getattr(args, "grades", None),
        query_count=len(table.rows) and _scored_query_count(table),
    )

    # Echo the per-row numbers so a CI log / terminal shows the real result.
    print("Ablation rows:")
    for row_cfg, metrics in table.rows:
        if metrics is None:
            print(f"  {row_cfg.name:16s} pending ({row_cfg.pending_reason})")
        else:
            print(
                f"  {row_cfg.name:16s} R@5={metrics['recall_at_5']:.3f} "
                f"R@10={metrics['recall_at_10']:.3f} MRR={metrics['mrr']:.3f} "
                f"nDCG@10={metrics['ndcg_at_10']:.3f}"
            )

    if args.print_only:
        print()
        print(ablation_section)
        return 0

    _merge_into_doc(out_path, ablation_section)
    print(f"\nWrote ablation section to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
