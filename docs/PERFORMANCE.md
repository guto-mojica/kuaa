# Retrieval performance

This page documents the retrieval benchmark harness. It does not commit a
current headline latency number because the benchmark depends on local
per-film library artifacts and hardware. Re-run the harness against the final
demo bundle before publishing README or portfolio numbers.

## Benchmark command

```bash
uv run python scripts/bench_retrieval.py --n 100 --k 50 --film <film_slug>
```

The script writes:

- JSON samples and summary stats to `data/perf/bench_results.json`;
- a Markdown report to `docs/PERFORMANCE.md`.

Both outputs are rebuilt on every run.

## What is timed

The benchmark measures one already-indexed film and reports p50, p95, p99,
mean, and max latency for `clip`, `bm25`, and `hybrid`.

`clip` uses the production `kuaa.search.clip.search_text` path.

`bm25` uses `BM25Index.query(query, top_k=raw_k)`.

`hybrid` mirrors `kuaa.search.hybrid.search_hybrid` and records four
sequential stages:

- `clip_best_row`: `_best_row_by_sid_from_embeddings(index, query)`;
- `clip_search`: `search_text(index, query, ..., raw_k, min_similarity)`;
- `bm25_query`: `BM25Index.query(query, top_k=raw_k)`;
- `rrf_materialize`: weighted RRF fusion plus `_fused_to_dataframe(...)`.

The current hybrid dispatcher performs two CLIP text encodes per query:
`clip_best_row` encodes once for best-keyframe backfill, and `clip_search`
encodes again through `search_text`. The benchmark keeps that behavior so the
numbers reflect production rather than a manually optimized approximation.

## Interpreting results

The first five hybrid queries are warm-up and are discarded. This primes the
CLIP model, BM25 score buffers, and lazy imports before timed samples begin.

The result payload records hardware, GPU/CUDA availability, selected device,
film slug, scene count, CLIP vector count, BM25 document count, `top_k`,
`raw_k`, `min_similarity`, RRF weights, and every per-query timing sample.

Do not quote old numbers after changing retrieval code, model backend,
embedding artifacts, `top_k`, RRF settings, or hardware. Re-run the command and
commit the regenerated report only when the artifact bundle being measured is
the one being released.
