#!/usr/bin/env python3
"""Recompute EN-vs-PT aggregates restricted to the 'evaluable subset'.

Evaluable = the 13 queries whose relevant_scene_ids include at least one scene
that lives in the indexed corpus (scenes 1-7 for the demo bundle). The other
11 queries point at scenes 8-13 which do not exist in the index, so every
retriever scores 0 on them — they only dilute the aggregate.

The published EVALUATION_RESULTS.md headline uses the same n=13 subset.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

INDEXED_SCENES = {"1", "2", "3", "4", "5", "6", "7"}
RETRIEVERS = ("clip", "bm25", "hybrid")


def _by_qid(payload):
    return {q["id"]: q for q in payload.get("queries", [])}


def _is_evaluable(q):
    rels = {str(s) for s in q.get("relevant_scene_ids", [])}
    return bool(rels & INDEXED_SCENES)


def _mean(rows, key):
    vals = [r["metrics"][key] for r in rows]
    return sum(vals) / len(vals) if vals else 0.0


def _summarize(dir_path: Path):
    out = {}
    for ret in RETRIEVERS:
        path = dir_path / f"{ret}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        evaluable = [q for q in payload["queries"] if _is_evaluable(q)]
        out[ret] = {
            "n": len(evaluable),
            "recall_at_5": _mean(evaluable, "recall_at_5"),
            "recall_at_10": _mean(evaluable, "recall_at_10"),
            "mrr": _mean(evaluable, "reciprocal_rank"),
            "ndcg_at_10": _mean(evaluable, "ndcg_at_10"),
        }
    return out


def main(argv):
    if len(argv) < 3:
        print(f"usage: {argv[0]} <en_dir> <pt_dir> [openclip_en_dir]", file=sys.stderr)
        return 2
    en = _summarize(Path(argv[1]))
    pt = _summarize(Path(argv[2]))
    extra = _summarize(Path(argv[3])) if len(argv) > 3 else None

    print("# Evaluable-subset aggregate (n=13 queries, scenes ∩ index ≥ 1)")
    print()
    if extra:
        print("| Retriever | metric | OClip-EN | SigLIP-EN | SigLIP-PT | Δ PT-EN (SigLIP) |")
        print("| --- | --- | ---: | ---: | ---: | ---: |")
    else:
        print("| Retriever | metric | EN | PT | Δ PT−EN |")
        print("| --- | --- | ---: | ---: | ---: |")
    for ret in RETRIEVERS:
        for mk in ("recall_at_5", "recall_at_10", "mrr", "ndcg_at_10"):
            en_v = en[ret][mk]
            pt_v = pt[ret][mk]
            d = pt_v - en_v
            sign = "+" if d > 0 else ("−" if d < 0 else " ")
            if extra:
                ocl_v = extra[ret][mk]
                print(
                    f"| {ret} | {mk} | {ocl_v:.3f} | {en_v:.3f} | {pt_v:.3f} | "
                    f"{sign}{abs(d):.3f} |"
                )
            else:
                print(f"| {ret} | {mk} | {en_v:.3f} | {pt_v:.3f} | {sign}{abs(d):.3f} |")
    print()
    print(f"Queries counted: clip={en['clip']['n']}, bm25={en['bm25']['n']}, hybrid={en['hybrid']['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
