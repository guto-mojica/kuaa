

<!-- ABLATION START -->

## Retriever-variant ablation, human-graded (SigLIP2 default)

**Run date:** 2026-10-29 — `scripts/run_ablation.py` (with-rerank (cross-encoder live), seed=0, `--grades corpus01`).
**Query set:** `corpus01_queries.yaml` — 56 text queries, each scored against the grades recorded for it in run `corpus01`.

Every row retrieves through the library-wide path the app serves (`kuaa.search.aggregate`), which is also the path the graded pool was drawn from — so each row is scored on candidates a human actually judged rather than on the pool's leftovers.

**Human-validated methodology.** Labels are human-validated (run corpus01, n=2436 grades). Every row is scored on the same query set against the same human grades, so the comparison is apples-to-apples.

**Corpus.** Chronopolis 1982 — 1230 keyframes indexed, 56 text queries.
**Common query set.** 56 text queries.

| Retriever | Labels | Recall@5 | Recall@10 | MRR | nDCG@10 |
| --- | --- | ---: | ---: | ---: | ---: |
| clip | graded | 0.183 | 0.350 | 0.961 | 0.644 |
| bm25 | graded | 0.116 | 0.217 | 0.722 | 0.381 |
| hybrid | graded | 0.178 | 0.344 | 0.920 | 0.677 |
| hybrid_no_metadata | graded | 0.178 | 0.345 | 0.938 | 0.675 |
| hybrid_rerank | graded | 0.188 | 0.355 | 0.932 | 0.676 |

> **hybrid_no_metadata.** Identical to `hybrid` except the exact-lexical metadata leg (tags / descriptions / detected objects) is disabled (`metadata_w=0`) — the delta to the `hybrid` row isolates that signal's contribution.
> **hybrid_rerank.** The bge-reranker-v2-m3 cross-encoder applied on top of the `hybrid` leg. It reads a first stage widened to the reranker's input window (`retrieval.reranker.top_k_in`), reorders it, and cuts back to `k` — so it can promote a candidate `hybrid` truncated away, and it does add rows to the pool. Compare it to the `hybrid` row it sits on, not to `clip`.

**Reading the numbers:**

- **Relevance is pooled, so recall is recall over judged scenes.** A candidate no retriever in the pool proposed was never shown to a grader and counts as irrelevant. That is the standard pooled-evaluation contract; it means these numbers compare the pooled variants to each other, and do not estimate absolute recall over the corpus.

- **`hybrid` vs `hybrid_no_metadata` isolates the exact-lexical leg.** The two rows differ only in `metadata_w`, so the delta is that signal's contribution and nothing else.

- **The rerank row sits on the `hybrid` row's candidates.** It widens the first stage to the reranker's input window, reorders, and cuts back — compare it to `hybrid`, not to `clip`.

- **PT/EN is a design axis of the query set, not of this table.** The pt-NN / en-NN pairs make the gap measurable per query; `scripts/check_pt_parity.py` is the surface that reports it.

Reproduce:

```bash
uv run python scripts/run_ablation.py \
  --queries data/eval/corpus01_queries.yaml --library-dir data/library \
  --seed 0 --with-rerank --grades corpus01 \
  --out docs/EVALUATION_RESULTS.md
```

<!-- ABLATION END -->

## Per-query breakdown of the graded ablation

The same five variants, the same grades and the same ranking path as the table
above, scored one query at a time and then grouped. Every cell is nDCG@10. The
groups hold 16–28 queries, so read differences of a few hundredths as noise:
single queries swing by ±0.3 or more.

| Group | n | clip | bm25 | hybrid | hybrid_no_metadata | hybrid_rerank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PT | 28 | 0.611 | 0.313 | 0.666 | 0.656 | 0.672 |
| EN | 28 | 0.677 | 0.448 | 0.689 | 0.694 | 0.680 |
| short (1–2 words) | 20 | 0.702 | 0.405 | 0.724 | 0.733 | 0.654 |
| mid (3–5 words) | 20 | 0.567 | 0.272 | 0.610 | 0.596 | 0.666 |
| long (6+ words) | 16 | 0.667 | 0.485 | 0.702 | 0.701 | 0.715 |

- **The PT/EN gap is small for `hybrid`.** Over the 28 `pt-NN`/`en-NN` pairs the
  mean PT−EN difference is −0.023 for `hybrid`, −0.065 for `clip` and −0.136 for
  `bm25`. Fusion absorbs most of the gap that each leg has alone.
- **The metadata leg does not help short object queries.** It exists for them,
  but `hybrid_no_metadata` is not worse there (0.733 against 0.724). Across all
  56 queries the two variants differ by more than 0.05 on only 16, split 8 wins
  and 8 losses.
- **The reranker nets to zero by cancelling.** Against `hybrid` it wins on 18
  queries, loses on 23 and ties on 15, with swings from +0.64 (`en-22`) to −0.31
  (`en-06`). It does worse on short queries and better on mid-length ones; with
  20 queries a group, that split is not established.
- **`hybrid` against `clip`:** 21 wins, 16 losses, 19 ties. Hybrid's gain is
  carried by a few queries (`en-13` +0.61, `pt-13` +0.45).
- **The weakest queries are compositional or about small objects:** `fish`
  (best variant 0.33), `silhueta contra fundo claro` (0.40), `text overlaid on
  the image` (0.54), `symmetrical composition with a central vanishing point`
  (0.59). These are the cases where a caption loses what the frame shows.
- **Recall@10 barely discriminates.** The median query has 27 relevant scenes,
  so Recall@10 cannot exceed about 0.37. Use nDCG@10 and MRR to compare
  variants, or raise `k` before drawing conclusions from recall.
- **MRR is 0.96 for `clip`.** The first hit is almost always relevant; what
  remains to gain is depth in the top 10.
