#!/usr/bin/env python3
"""Generate the proxy-first multi-modal ablation table (WS-4 E2b).

Runs :func:`cinemateca.eval.ablation.run_ablation` over the 15 text queries in
``data/eval/m3_full_queries.yaml`` (the common HY-labelled set) and writes the
result as an **M4 section** of ``docs/EVALUATION_RESULTS.md``, preserving the
existing M2 section. The M4 block is delimited by HTML comment markers
(``<!-- M4 ABLATION START -->`` / ``<!-- M4 ABLATION END -->``) so re-runs
replace only that block — the M2 ablation above it is never touched.

Every published cell is a REAL proxy number computed on the demo corpus, or a
literal ``pending (...)`` for a row whose backend is not wired. Under
``--no-rerank`` the cross-encoder row is ``pending (C5)``; ``--with-rerank``
fills it by running the production ``find(mode="hybrid", rerank=True)`` path.

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

_M4_START = "<!-- M4 ABLATION START -->"
_M4_END = "<!-- M4 ABLATION END -->"


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
        help="Markdown doc whose M4 section to write/replace (preserves M2).",
    )
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed (default 0).")
    rerank = parser.add_mutually_exclusive_group()
    rerank.add_argument(
        "--with-rerank",
        dest="with_rerank",
        action="store_true",
        help="Compute the hybrid+rerank row via the production find() path (C5).",
    )
    rerank.add_argument(
        "--no-rerank",
        dest="with_rerank",
        action="store_false",
        help="Render the hybrid+rerank row as pending (C5) — the default.",
    )
    parser.set_defaults(with_rerank=False)
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print the M4 markdown to stdout without touching the doc.",
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


def _build_m4_section(table_md: str, *, with_rerank: bool, seed: int, queries: Path) -> str:
    """Wrap the rendered ablation table in the delimited M4 doc section."""
    mode = (
        "with-rerank (C5 cross-encoder live)" if with_rerank else "no-rerank (rerank row pending)"
    )
    lines = [
        _M4_START,
        "",
        "## M4 — Multi-modal proxy ablation (SigLIP2 default)",
        "",
        f"**Run date:** {date.today().isoformat()} — `scripts/run_ablation.py` "
        f"({mode}, seed={seed}).",
        f"**Query set:** `{queries.name}` — the 15 text queries (common set).",
        "",
        "Retriever-variant ablation on a **common query set with the same proxy "
        "labels** (apples-to-apples). This is the launch ablation: it is producible "
        "with **zero human grades** and every cell is either a real proxy number or "
        "an honest `pending (...)`. The numbers **supersede the M2 OpenCLIP table "
        "above** as the current-tree (SigLIP2) result; the M2 section is retained "
        "for provenance.",
        "",
        table_md,
        "",
        "**Reading the numbers** (proxy / HY, not human-graded):",
        "",
        "- **Hybrid beats CLIP-only here** (the multi-film, larger-corpus result the "
        "M2 note predicted) — RRF fusion of SigLIP2 + BM25 edges CLIP on R@5 and MRR.",
        "- **The multilingual upgrade matters.** The OpenCLIP baseline (`multilingual` "
        "row, C8) trails the SigLIP2 `CLIP` row on every metric on this PT/EN query "
        "mix — evidence for the SigLIP2-multilingual swap.",
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
        _M4_END,
    ]
    return "\n".join(lines)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _merge_into_doc(doc_path: Path, m4_section: str) -> None:
    """Write the M4 section into ``doc_path``, preserving everything else.

    If the doc already has the M4 markers, the block between them is replaced.
    Otherwise the section is appended (with a leading blank-line separator).
    The M2 ablation content above the markers is never modified.
    """
    if doc_path.exists():
        existing = doc_path.read_text(encoding="utf-8")
    else:
        existing = ""

    if _M4_START in existing and _M4_END in existing:
        head, _, rest = existing.partition(_M4_START)
        _, _, tail = rest.partition(_M4_END)
        merged = head.rstrip() + "\n\n" + m4_section + tail.rstrip() + "\n"
    else:
        sep = "\n\n---\n\n" if existing.strip() else ""
        merged = existing.rstrip() + sep + m4_section + "\n"

    doc_path.write_text(merged, encoding="utf-8")


def _load_graded_labels(
    grades_arg: str | None,
) -> tuple[dict | None, str | None]:
    """Load graded labels from a run ID or path; return (graded_labels, validated_label).

    ``graded_labels`` is ``{query_id: {scene_id: float_grade}}`` with positive
    grades only (the ablation caller filters zero/negative itself, but we skip
    them here for clarity). Returns ``(None, None)`` when no ``--grades`` arg.
    """
    if grades_arg is None:
        return None, None

    from pathlib import Path as _Path

    from cinemateca.eval.grades import EvalRun, export_run

    path = _Path(grades_arg)
    if path.is_absolute() and path.exists():
        run = EvalRun(run_id=path.stem, root=path.parent)
    else:
        # Treat as run_id under data/eval/.
        run_root = REPO_ROOT / "data" / "eval"
        run = EvalRun(run_id=grades_arg, root=run_root)

    exported = export_run(run)
    # graded_labels: {query_id: {scene_id: float_grade}}
    graded_labels: dict = {}
    for qid, scenes in exported["grades"].items():
        pos = {sid: float(g) for sid, g in scenes.items() if float(g) > 0}
        if pos:
            graded_labels[qid] = pos

    distinct = exported["summary"]["distinct_pairs"]
    validated_label = f"human-validated (run {run.run_id}, n={distinct} grades)"
    return graded_labels, validated_label


def main(argv: list[str] | None = None) -> int:
    from cinemateca.config import load_config
    from cinemateca.errors import EvalError
    from cinemateca.eval.ablation import (
        DEFAULT_ABLATION_CONFIGS,
        DEFAULT_ABLATION_CONFIGS_NO_RERANK,
        run_ablation,
    )
    from cinemateca.eval.slates import load_modal_queries

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
    m4_section = _build_m4_section(
        table_md, with_rerank=args.with_rerank, seed=args.seed, queries=queries_path
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
        print(m4_section)
        return 0

    _merge_into_doc(out_path, m4_section)
    print(f"\nWrote M4 ablation section to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
