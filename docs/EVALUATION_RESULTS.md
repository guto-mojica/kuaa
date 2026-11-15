

<!-- ABLATION START -->

## Retriever-variant ablation, human-graded (SigLIP2 default)

**Run date:** 2026-11-10 — `scripts/run_ablation.py` (with-rerank (cross-encoder live), seed=0, `--grades corpus01`).
**Query set:** `corpus01_queries.yaml` — 56 text queries, each scored against the grades recorded for it in run `corpus01`.

Every row retrieves through the library-wide path the app serves (`kuaa.search.aggregate`), which is also the path the graded pool was drawn from — so each row is scored on candidates a human actually judged rather than on the pool's leftovers.

**Human-validated methodology.** Labels are human-validated (run corpus01, n=1288 grades). Every row is scored on the same query set against the same human grades, so the comparison is apples-to-apples.

**Corpus.** Whole library — 11 films, 3018 keyframes indexed, 56 text queries.
**Common query set.** 56 text queries.

| Retriever | Labels | Recall@5 | Recall@10 | MRR | nDCG@10 |
| --- | --- | ---: | ---: | ---: | ---: |
| clip | graded | 0.267 | 0.505 | 0.961 | 0.669 |
| bm25 | graded | 0.168 | 0.307 | 0.722 | 0.389 |
| hybrid | graded | 0.262 | 0.499 | 0.920 | 0.701 |
| hybrid_no_metadata | graded | 0.262 | 0.498 | 0.938 | 0.699 |
| hybrid_rerank | graded | 0.273 | 0.512 | 0.923 | 0.696 |

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
| PT | 28 | 0.633 | 0.320 | 0.685 | 0.676 | 0.688 |
| EN | 28 | 0.705 | 0.459 | 0.716 | 0.722 | 0.704 |
| short (1–2 words) | 20 | 0.728 | 0.408 | 0.749 | 0.758 | 0.673 |
| mid (3–5 words) | 20 | 0.592 | 0.288 | 0.635 | 0.621 | 0.686 |
| long (6+ words) | 16 | 0.691 | 0.494 | 0.722 | 0.721 | 0.737 |

Wins and losses below count a query only when the two variants differ by more
than 0.05; anything closer is a tie.

- **The PT/EN gap is small for `hybrid`.** Over the 28 `pt-NN`/`en-NN` pairs the
  mean PT−EN difference is −0.031 for `hybrid`, −0.072 for `clip` and −0.139 for
  `bm25`. Fusion absorbs most of the gap that each leg has alone.
- **The metadata leg does not help short object queries.** It exists for them,
  but `hybrid_no_metadata` is not worse there (0.758 against 0.749). Across all
  56 queries the two variants differ by more than 0.05 on only 16, split 8 wins
  and 8 losses.
- **The reranker nets to slightly below `hybrid`.** Against `hybrid` it wins on
  17 queries, loses on 24 and ties on 15, with swings from +0.66 (`en-22`) to
  −0.33 (`en-06`). It does worse on short queries and better on mid-length ones;
  with 20 queries a group, that split is not established.
- **`hybrid` against `clip`:** 21 wins, 16 losses, 19 ties. Hybrid's gain is
  carried by a few queries (`en-13` +0.61, `pt-13` +0.45).
- **The weakest queries are about silhouette and composition:** `silhueta
  contra fundo claro` (best variant 0.43), `duas pessoas conversando` (0.62),
  and `symmetrical composition with a central vanishing point` (0.63 in English,
  0.64 in Portuguese). These are the cases where a caption loses what the frame
  shows.
- **Recall@10 discriminates weakly.** The median query has 19 relevant scenes
  in the pool, so Recall@10 cannot exceed about 0.53. Use nDCG@10 and MRR to
  compare variants, or raise `k` before drawing conclusions from recall.
- **MRR is 0.96 for `clip`.** The first hit is almost always relevant; what
  remains to gain is depth in the top 10.
- **Only grades on the current pool are scored.** The run's `.jsonl` also holds
  1148 judgments from an earlier pool. They are left out, because a positive one
  would count as a relevant scene that no current variant was pooled on, which
  lowers Recall and nDCG for every row.
- **The pool predates the index the rows now rank.** It was generated on
  2026-09-01. On this run, 37 of `bm25`'s 494 top-10 hits are not in the pool,
  and 11 of `hybrid_rerank`'s 544; `hybrid` and `hybrid_no_metadata` have 2 each
  and `clip` none. An unpooled hit was never judged and scores as irrelevant, so
  `bm25` and `hybrid_rerank` are understated by up to that share. Regenerating
  the pool and grading the new candidates would close it.
