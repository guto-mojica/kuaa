# Evaluation

Repeatable measurement harness for the public demo. Shows retrieval quality and
human-in-the-loop metadata value with stable files and commands.

## Scope

This harness covers three things:

- a reviewed query dataset for the public archive demo;
- deterministic retrieval metrics over an existing retrieval index;
- correction statistics comparing generated tags with manual annotations.

Implemented files:

- `data/eval/archive_demo_queries.yaml`
- `src/kuaa/eval/`
- `scripts/run_eval.py`
- `tests/test_eval_metrics.py`
- `tests/test_annotation_stats.py`

The evaluation code is intentionally file-based so it can run against the
public demo bundle or a local private collection without uploading video,
frames, metadata, or embeddings.

## Artifact Prerequisite

The repository tracks query labels and evaluation code, but it does not track
the runtime media, keyframes, metadata, or embedding artifacts. Prepare or
extract a demo bundle before running the commands below:

```bash
uv run python scripts/prepare_demo.py --download
```

## Commands

`scripts/run_eval.py` evaluates a KUAA index (CLIP / BM25 / hybrid). The
current CLI resolves the artifacts to score in one of two ways:

- Pass `--film-slug <slug>` to evaluate a per-film library at
  `data/library/<slug>/` — this overrides `cfg.paths.{embeddings,metadata,
  frames}_dir` at runtime and is the normal way to score a single film in the
  current per-film library layout.
- Pass `--config <path>` (e.g. `config/demo.yaml`) to evaluate whatever flat
  `paths.*` the config points at, with no `--film-slug` override — this is how
  the demo bundle under `data/demo/runtime/` is scored, since it is not
  registered as a per-film library entry.

There is no `--library-dir` flag on `run_eval.py` (that flag exists only on
`scripts/run_ablation.py`, described below, which scans a per-film library
root for its multi-modal ablation).

### run_eval.py — retrieval scoring

Single-mode run against a per-film library (SigLIP2 index):

```bash
# Text modality — single retriever, per-film library:
uv run python scripts/run_eval.py \
  --queries data/eval/m3_full_queries.yaml \
  --retriever clip   --top-k 10 --film-slug jeca_tatu
uv run python scripts/run_eval.py \
  --queries data/eval/m3_full_queries.yaml \
  --retriever bm25   --top-k 10 --film-slug jeca_tatu
uv run python scripts/run_eval.py \
  --queries data/eval/m3_full_queries.yaml \
  --retriever hybrid --top-k 10 --film-slug jeca_tatu \
  --sem-weight 0.5 --bm25-weight 0.5 --k-rrf 60
```

The single-mode runner writes `summary.json` + `report.md` (and
annotation-correction stats when `scene_tags.json` and
`manual_annotations.json` are present).

Full ablation (CLIP / BM25 / hybrid + optional k_rrf sweep) against the demo
bundle:

```bash
uv run python scripts/run_eval.py --config config/demo.yaml \
  --queries data/eval/archive_demo_queries.yaml \
  --all-modes --top-k 10 \
  --k-rrf-sweep "10,30,60,100"
```

The all-modes runner writes `clip.json`, `bm25.json`, `hybrid.json`,
`krrf_sweep.json`, plus a human-readable `comparison.md` and the
machine-readable `comparison.json`.

#### Multi-modal scoring (`--modality`)

`scripts/run_eval.py` scores the text, image, and rhyme retrieval modalities.
Pass `--film-slug` (per-film library) or `--config` (flat demo layout) for the
artifacts to score. Non-text modalities use proxy / known-item relevance:

| `--modality` | Retrieval backend | Default relevance method |
|---|---|---|
| `text` (default) | CLIP / BM25 / hybrid | YAML hypotheses (HY) |
| `image` | CLIP `find` (image query) | Known-item (KI) — anchor scene from frame filename |
| `rhyme` | Cross-film `find_rhymes` | Known-item (KI) — the anchor scene |
| `all` | All modalities, one report each | Per-modality as above |

**Image KI is self-retrieval-only.** The correct answer for an `image` query
is the exact scene its keyframe was cropped from. A high score proves the
encoder can find its own source frame — not that it generalizes to other,
semantically similar scenes. Treat it as a sanity floor, not a quality proof;
real cross-film matching needs HY labels or curator grading.

```bash
# Score every modality at once (per-film library, SigLIP2):
uv run python scripts/run_eval.py \
  --queries data/eval/m3_full_queries.yaml \
  --modality all --film-slug jeca_tatu \
  --output-dir data/eval/m3-run-1/

# Score a single non-text modality:
uv run python scripts/run_eval.py \
  --queries data/eval/m3_full_queries.yaml \
  --modality image --film-slug jeca_tatu \
  --output-dir data/eval/m3-run-1/
```

### kuaa eval slate — generate grading slates

Pre-stage candidate slates for the `/eval` grading UI. Writes
`data/eval/<run>.queries.json` in the same format `kuaa eval seed`
produces, so the grading page renders real retrieved candidates rather than
placeholder rows:

```bash
uv run kuaa eval slate \
  --queries data/eval/m3_full_queries.yaml \
  --run m3-run-1 --root data/eval \
  --modality all       # or text | image | rhyme
```

### kuaa eval export — export curator grades

After a grading session, export the grade JSONL for downstream use:

```bash
# JSON export (per-query, last-write-wins per (query, scene) pair):
uv run kuaa eval export \
  --run m3-run-1 --root data/eval --format json

# CSV export (query_id,scene_id,grade rows):
uv run kuaa eval export \
  --run m3-run-1 --root data/eval --format csv
```

### run_ablation.py — multi-retriever proxy ablation

Produces a multi-retriever ablation table comparing retrievers across
modalities. Runs proxy-first (zero human grades required) and upgrades to
human-validated when `--grades` is passed with a completed curator export:

```bash
# Proxy run (HY labels from YAML):
uv run python scripts/run_ablation.py \
  --queries data/eval/m3_full_queries.yaml \
  --library-dir data/library \
  --seed 0 --with-rerank \
  --out docs/EVALUATION_RESULTS.md

# Human-validated run (upgrades the table in-place once curator grades land):
uv run python scripts/run_ablation.py \
  --queries data/eval/m3_full_queries.yaml \
  --library-dir data/library \
  --seed 0 --with-rerank \
  --grades m3-run-1 \
  --out docs/EVALUATION_RESULTS.md
```

`--library-dir` (unlike `run_eval.py`'s `--film-slug`) points at the per-film
library root and lets the ablation scan every registered film. Headline
numbers and per-query failure patterns are tracked in separate evaluation-
results and failure-analysis docs. Re-run the metrics against the final
artifact bundle before publishing numeric claims.

## Query Schema

The query file is YAML:

```yaml
dataset: archive_demo
version: 1
source:
  name: "The Great Train Robbery"
  identifier: "loc:00694220"
queries:
  - id: q001
    text: "bandits entering a train car"
    intent: "Object/action search"
    relevant_scene_ids: [4, 5]
    relevance:
      "4": 3
      "5": 2
    negative_scene_ids: [1]
    notes: "Higher grade for close, visually obvious matches."
```

Required fields:

- `dataset`
- `version`
- `queries[].id`
- `queries[].text`
- `queries[].relevant_scene_ids`

Optional fields:

- `source`
- `queries[].intent`
- `queries[].relevance`
- `queries[].negative_scene_ids`
- `queries[].notes`

`relevant_scene_ids` may contain integers or strings. Metric code canonicalizes
scene ids before comparison, matching the search and annotation layers.

## Retrieval Metrics

The report computes:

- `Recall@5`: mean per-query fraction of expected scenes found in the top five.
- `Recall@10`: mean per-query fraction of expected scenes found in the top ten.
- `MRR`: mean reciprocal rank of the first relevant result.
- `nDCG@10`: graded relevance quality when `relevance` is present.

For ungraded queries, all relevant scenes receive grade `1`. Queries without
relevant scenes are rejected by validation instead of being silently skipped.

## Annotation Stats

Manual annotations are stored in `manual_annotations.json` as scene-to-tags.
Generated tags are stored in `scene_tags.json` as tag-to-scenes. The stats
compare the two views after normalization and report:

- scenes with manual annotations;
- generated tags accepted by manual annotations;
- human-added tags not present in generated metadata;
- generated tags omitted by manual annotations;
- correction rate.

This is a proxy metric. It measures tag agreement and curator additions; it
does not prove that every unaccepted generated tag is wrong.

## Headline Numbers

Current retrieval results (ablation table, per-query breakdown) are tracked in
a separate evaluation-results document, published once curator grading is
complete.

## Release Notes

The query set is only as stable as the public artifact bundle. If scene
detection settings change or the demo ZIP is regenerated, re-run the demo,
review `keyframes_metadata.json`, update `archive_demo_queries.yaml`, and commit
the refreshed labels with the release artifact checksum update.
