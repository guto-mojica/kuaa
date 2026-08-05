<!-- M4 ABLATION START -->

## M4 — Multi-modal proxy ablation (SigLIP2 default)

**Run date:** 2026-08-04 — `scripts/run_ablation.py` (no-rerank (rerank row pending), seed=0).
**Query set:** `m3_full_queries.yaml` — the 15 text queries (common set).

Retriever-variant ablation on a **common query set with the same proxy labels** (apples-to-apples). This is the launch ablation: it is producible with **zero human grades** and every cell is either a real proxy number or an honest `pending (...)`.

**Proxy methodology.** These are **proxy metrics**, not human-graded ground truth — they upgrade to human-validated numbers when curator grades land (WS-4 E5). Every row below is scored on a common query set with the **same** proxy labels, so the comparison is apples-to-apples. Proxy signals:

- **HY (Hypothesis)** — the maintainer's pre-curator `relevant_scene_ids` / `relevance` from the query file. Best-guess relevant scenes recorded before any grading session.
- **KI (Known-Item)** — the single anchor scene a query came from (image keyframe / rhyme anchor). Not used in this table.
- **PR (Pseudo-Relevance)** — a reference retriever's top-1 treated as relevant (relative agreement). Not used in this table.

**Corpus.** Jeca Tatu 1959 — 1344 keyframes indexed, 15 text queries.
**Common query set.** 15 text queries (m3_full) — all labelled **HY**.

| Retriever | Proxy | Recall@5 | Recall@10 | MRR | nDCG@10 |
| --- | --- | ---: | ---: | ---: | ---: |
| CLIP | HY | 0.067 | 0.089 | 0.080 | 0.077 |
| BM25 | HY | 0.111 | 0.156 | 0.119 | 0.110 |
| hybrid | HY | 0.111 | 0.111 | 0.120 | 0.094 |
| hybrid-metadata | HY | 0.111 | 0.111 | 0.120 | 0.094 |
| hybrid+rerank | HY | pending (C5) | pending (C5) | pending (C5) | pending (C5) |

> **hybrid-metadata.** Identical to `hybrid` except the exact-lexical metadata leg (tags / descriptions / detected objects) is disabled (`metadata_w=0`) — the delta to the `hybrid` row isolates that signal's contribution.
> **hybrid+rerank.** Rerank delta is measured on the production `find(mode="hybrid")` base (± the C5 bge-reranker-v2-m3 cross-encoder), which is a different hybrid implementation from the harness `hybrid` row above — compare the rerank row to the `find` hybrid base it sits on, not to the harness `hybrid` row.

**Reading the numbers** (proxy / HY, not human-graded):

- **Hybrid beats CLIP-only here** — RRF fusion of SigLIP2 + BM25 edges CLIP on R@5 and MRR.

Reproduce:

```bash
uv run python scripts/run_ablation.py \
  --queries data/eval/m3_full_queries.yaml --library-dir data/library \
  --seed 0 --no-rerank \
  --out docs/EVALUATION_RESULTS.md
```

<!-- M4 ABLATION END -->
