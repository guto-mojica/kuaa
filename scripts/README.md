# scripts/

Operational and analysis scripts. Two tiers:

## Critical (CI / release / operations — do not delete without updating callers)

| Script | Role | Invoked by |
|---|---|---|
| `check_loc_budget.py` | LOC budget gate (services ≤250, routes ≤150) | `.github/workflows/refactor-guards.yml` |
| `verify_fresh_run.sh` | Clean checkout → `uv sync` → boot → `/health` | T8 gate; run before a release |
| `run_eval.py` | Retrieval eval (clip/bm25/hybrid; `--all-modes`) | SETUP §7, WS-4 eval, EVALUATION_RESULTS |
| `bench_retrieval.py` | Latency p50/p95/p99 per retriever | WS-4 E6, T9 CI benchmark job |
| `verify_features.py` | E2E audio/fusion/reranker on real artefacts | release verification |
| `check_launch_package.py` | Zero-placeholder gate on public docs | D10 |
| `build_demo_bundle.py` | Deterministic demo ZIP | demo release |
| `prepare_demo.py` | Download/validate demo bundle | SETUP §7 |
| `freeze_eval_run.sh` | SHA256-tar eval grades for provenance | EVAL_PROTOCOL §7 |
| `migrate_flat_to_library.py` | One-shot v0.3 flat→per-film migration | historical, run-once |
| `ensure_gpu_llama.sh` | Rebuild CUDA llama-cpp after `uv sync` | GPU describer ops |

## Exploratory (private, ad-hoc analysis — safe to ignore in CI)

| Script | Role |
|---|---|
| `analyze_pt_gap.py` | EN-vs-PT retrieval gap analysis (SigLIP multilingual) |
| `analyze_pt_gap_evaluable.py` | Same, restricted to the n=13 evaluable subset |

Run any script with `uv run python scripts/<name>.py --help` (Python) or
`scripts/<name>.sh` (bash). All paths resolve relative to the repo root.
