#!/usr/bin/env python3
"""Compare EN vs PT retrieval runs for the SigLIP-multilingual gap analysis.

Reads ``per-mode.json`` outputs from two ``scripts/run_eval.py --all-modes`` runs
(one EN, one PT) and prints:

  * aggregate EN-vs-PT deltas per retriever (R@5/R@10/MRR/nDCG@10);
  * per-query rank deltas for the top relevant scene, sorted by absolute delta;
  * the queries that flipped from "found in top-K" to "not found" or vice versa.

Usage:

    uv run python scripts/analyze_pt_gap.py \
        --en-dir data/eval/reports/en_siglip \
        --pt-dir data/eval/reports/pt_siglip

Optional ``--en-openclip-dir`` adds a third column for a CLIP-EN baseline run
on the OpenClip backend (the reproduced published numbers).

This is a private measurement tool for §7 of PORTFOLIO_FIXES.md. Outputs go to
stdout; the caller copies what it needs into ``.notes/PT_GAP_FINDINGS.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

RETRIEVERS = ("clip", "bm25", "hybrid")
METRIC_KEYS = ("recall_at_5", "recall_at_10", "mrr", "ndcg_at_10")


def _load_mode(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing report: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _by_qid(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {q["id"]: q for q in payload.get("queries", [])}


def _best_rank_of_relevant(query_payload: dict[str, Any]) -> int | None:
    """Return the 1-indexed rank of the highest-graded relevant scene_id.

    Falls back to the first relevant scene_id that appears in the ranked list
    if a tie. ``None`` means none of the relevant scene_ids landed in top-K.
    """
    relevant = [str(s) for s in query_payload.get("relevant_scene_ids", [])]
    ranked = [str(s) for s in query_payload.get("ranked_scene_ids", [])]
    best: int | None = None
    for sid in relevant:
        if sid in ranked:
            r = ranked.index(sid) + 1
            if best is None or r < best:
                best = r
    return best


def _delta(en_val: float, pt_val: float) -> str:
    d = pt_val - en_val
    sign = "+" if d > 0 else ("−" if d < 0 else " ")
    return f"{sign}{abs(d):.3f}"


def _aggregate_table(en: dict, pt: dict, label_en: str = "EN", label_pt: str = "PT") -> str:
    lines = [
        "| Retriever | metric | "
        f"{label_en} | {label_pt} | Δ (PT−{label_en}) |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for ret in RETRIEVERS:
        en_mode = next((m for m in en["modes"] if m["retriever"] == ret), None)
        pt_mode = next((m for m in pt["modes"] if m["retriever"] == ret), None)
        if not en_mode or not pt_mode:
            continue
        for mk in METRIC_KEYS:
            en_v = en_mode["metrics"][mk]
            pt_v = pt_mode["metrics"][mk]
            lines.append(
                f"| {ret} | {mk} | {en_v:.3f} | {pt_v:.3f} | {_delta(en_v, pt_v)} |"
            )
    return "\n".join(lines)


def _rank_delta_table(en_payload: dict, pt_payload: dict) -> str:
    en_q = _by_qid(en_payload)
    pt_q = _by_qid(pt_payload)
    rows: list[tuple[str, str, str, int | None, int | None, int]] = []
    for qid, en_data in en_q.items():
        if qid not in pt_q:
            continue
        en_rank = _best_rank_of_relevant(en_data)
        pt_rank = _best_rank_of_relevant(pt_q[qid])
        # Treat "missing from top-K" as rank 11 for sorting only.
        en_for_sort = en_rank if en_rank is not None else 11
        pt_for_sort = pt_rank if pt_rank is not None else 11
        delta_abs = abs(pt_for_sort - en_for_sort)
        rows.append((qid, en_data["text"], pt_q[qid]["text"], en_rank, pt_rank, delta_abs))
    rows.sort(key=lambda r: -r[5])
    lines = [
        "| qid | EN text | PT text | EN rank | PT rank | |Δ| |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for qid, en_t, pt_t, en_r, pt_r, da in rows:
        lines.append(
            f"| `{qid}` | {en_t[:48]} | {pt_t[:48]} | "
            f"{en_r if en_r is not None else '—'} | "
            f"{pt_r if pt_r is not None else '—'} | {da} |"
        )
    return "\n".join(lines)


def _flips(en_payload: dict, pt_payload: dict) -> tuple[list[str], list[str]]:
    en_q = _by_qid(en_payload)
    pt_q = _by_qid(pt_payload)
    pt_only_misses: list[str] = []
    en_only_misses: list[str] = []
    for qid, en_data in en_q.items():
        if qid not in pt_q:
            continue
        en_rank = _best_rank_of_relevant(en_data)
        pt_rank = _best_rank_of_relevant(pt_q[qid])
        if en_rank is not None and pt_rank is None:
            pt_only_misses.append(qid)
        elif en_rank is None and pt_rank is not None:
            en_only_misses.append(qid)
    return pt_only_misses, en_only_misses


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--en-dir", required=True, help="Path to EN comparison.json's parent dir")
    p.add_argument("--pt-dir", required=True, help="Path to PT comparison.json's parent dir")
    p.add_argument(
        "--en-openclip-dir",
        default=None,
        help="Optional OpenClip-EN comparison dir for triangulation.",
    )
    args = p.parse_args(argv)

    en_dir = Path(args.en_dir)
    pt_dir = Path(args.pt_dir)
    en_cmp = _load_mode(en_dir / "comparison.json")
    pt_cmp = _load_mode(pt_dir / "comparison.json")
    en_clip = _load_mode(en_dir / "clip.json")
    pt_clip = _load_mode(pt_dir / "clip.json")
    en_hybrid = _load_mode(en_dir / "hybrid.json")
    pt_hybrid = _load_mode(pt_dir / "hybrid.json")
    en_bm25 = _load_mode(en_dir / "bm25.json")
    pt_bm25 = _load_mode(pt_dir / "bm25.json")

    print("# EN vs PT aggregate (SigLIP-multilingual default)")
    print()
    print(_aggregate_table(en_cmp, pt_cmp))
    print()

    if args.en_openclip_dir:
        ocl_cmp = _load_mode(Path(args.en_openclip_dir) / "comparison.json")
        print("# EN reproduced on OpenClip ViT-B/32 (published baseline) — vs SigLIP-EN")
        print()
        print(_aggregate_table(ocl_cmp, en_cmp, label_en="OClip-EN", label_pt="SigLIP-EN"))
        print()

    print("## Per-query rank delta (highest-graded relevant scene) — clip mode")
    print()
    print(_rank_delta_table(en_clip, pt_clip))
    print()
    print("## Per-query rank delta — hybrid mode")
    print()
    print(_rank_delta_table(en_hybrid, pt_hybrid))
    print()
    print("## Per-query rank delta — bm25 mode")
    print()
    print(_rank_delta_table(en_bm25, pt_bm25))
    print()

    print("## In-top-K flips (clip mode)")
    pt_miss, en_miss = _flips(en_clip, pt_clip)
    print(f"  PT lost relevant from top-10 (had it in EN): {pt_miss or '—'}")
    print(f"  PT gained relevant in top-10 (missed in EN): {en_miss or '—'}")
    print()
    print("## In-top-K flips (hybrid mode)")
    pt_miss, en_miss = _flips(en_hybrid, pt_hybrid)
    print(f"  PT lost relevant from top-10 (had it in EN): {pt_miss or '—'}")
    print(f"  PT gained relevant in top-10 (missed in EN): {en_miss or '—'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
