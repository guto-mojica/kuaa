# Evaluation Data

Small, reviewable evaluation datasets. Files here are source data and should
stay light enough to read in a pull request.

## Three files, three roles

The names do not make the distinction obvious, so:

| File | Role | Written by |
|---|---|---|
| `<name>.yaml` | The **queries** | A human, by hand |
| `<run>.queries.json` | The **pooled candidates** to judge | `kuaa eval slate` |
| `<run>.jsonl` | The **judgments** | The `/eval` UI, one row per keypress |

The `.jsonl` is the ground truth and outlives everything else. Any retriever —
including ones that do not exist yet — can be scored against it, which is the
whole reason the candidates are pooled rather than taken from one retriever.

## Query file schema

Top-level `queries:` list. Per entry:

| Field | Applies to | Meaning |
|---|---|---|
| `id` | all | Stable string (`text-01`, `image-01`, `rhyme-01`) |
| `query_type` | all | `text` \| `image` \| `rhyme` |
| `text` | text | The query string |
| `image_path` | image | Repo-relative keyframe path; must exist on disk |
| `anchor` | rhyme | `<slug>/<scene_id>` |
| `lang` | all | `pt` \| `en` |
| `notes` | all | Curator-facing context |

Validated by `kuaa.eval.slates.load_modal_queries`, which raises `EvalError` on
a missing image, an unparseable anchor, or an unknown `query_type`.

**Do not add `relevant_scene_ids` / `relevance` to a query set meant for
grading.** Those fields hold pre-annotation *hypotheses*, and a label authored
from what the system already returned is only valid for the system that
returned it — it cannot reward a fix that surfaces scenes the old pool never
showed. They exist for `scripts/run_eval.py`, which still requires them.
Grading supersedes them.

## Authoring a query set

Write the queries **after** the corpus is indexed, against the films that are
actually there. Queries written before seeing the library reference scenes that
do not exist and degrade into hypotheses.

Cover both ends of the query-length range. `hybrid_metadata_w` tapers with
query length (`kuaa.retrieval.hybrid.effective_metadata_w`): the exact-lexical
leg earns a majority share on short object queries and actively hurts long
natural-language ones. A set of only long queries — which is what the previous
`m3_full` set was, 4–7 terms each — never exercises the end the leg exists for.

Cover Portuguese properly. The BM25 corpus is English (the describer prompts
are English) and `kuaa.retrieval.bilingual` expands it at index time. PT is the
primary use case and the axis the old labels were blind to.

## Generating and grading

```bash
uv run kuaa eval slate --queries data/eval/<name>.yaml \
  --run <run> --root data/eval --modality all --k 10

EVAL_ADMIN_TOKEN=dev uv run kuaa serve     # then open /eval?token=dev
```

Set a `grader` cookie to your name before grading — it is the field the
inter-annotator κ reads to tell two graders apart, and it defaults to `anon`.

Rows arrive shuffled with `score` blanked. That is deliberate: position would
otherwise be the retriever's opinion, and a grader who reads it grades the
system instead of the scene. Each row carries a `pool` key recording which
variant proposed it at which rank — the template ignores it, the scorer needs
it.

## Archive Demo

`archive_demo_queries.yaml` is the M2 starter query set for the public demo
configured by `config/demo.yaml`, anchored to the expected demo scene order for
*The Great Train Robbery* (1903). It carries hypothesis labels because
`scripts/run_eval.py` requires them.

```bash
uv run python scripts/run_eval.py \
  --config config/demo.yaml \
  --queries data/eval/archive_demo_queries.yaml \
  --output-dir data/eval/reports
```

Reports under `data/eval/reports/` are runtime artifacts. Commit only curated
summaries meant to become release documentation.
