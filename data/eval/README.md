# Evaluation Data

This directory contains small, reviewable evaluation datasets. Files here are
source data and should stay lightweight enough to review in a pull request.

## Archive Demo

`archive_demo_queries.yaml` is the M2 starter query set for the public LOC demo
configured by `config/demo.yaml`.

The scene labels are anchored to the expected public-demo scene order for *The
Great Train Robbery* (1903). Before a final release, regenerate or download the
demo artifact bundle, inspect `data/demo/runtime/metadata/keyframes_metadata.json`,
and refresh the relevant scene ids if scene detection settings changed.

Run:

```bash
uv run python scripts/run_eval.py \
  --config config/demo.yaml \
  --queries data/eval/archive_demo_queries.yaml \
  --output-dir data/eval/reports
```

Generated reports under `data/eval/reports/` are runtime artifacts. Commit only
curated summaries when they are meant to become release documentation.
